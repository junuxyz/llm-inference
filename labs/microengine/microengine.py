from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class RequestState(StrEnum):
    """Lifecycle state for one generation request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(slots=True)
class Request:
    """Per-request state tracked across admission, prefill, and decode."""

    request_id: str
    prompt_text: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None
    output_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_text:
            raise ValueError("prompt_text must not be empty")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

    @property
    def is_finished(self) -> bool:
        return self.state is RequestState.FINISHED

    def start(self) -> None:
        if self.state is not RequestState.WAITING:
            raise RuntimeError(f"Cannot start request in state={self.state}")
        self.state = RequestState.RUNNING

    def finish(self) -> None:
        self.state = RequestState.FINISHED

    def record_token(self, token_id: int) -> None:
        if self.state is RequestState.FINISHED:
            raise RuntimeError("Cannot append token to a finished request")

        self.output_ids.append(int(token_id))
        if len(self.output_ids) >= self.max_new_tokens:
            self.finish()
            return
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self.finish()


class RequestQueue:
    """FIFO admission queue for waiting requests."""

    def __init__(self) -> None:
        self._queue: deque[Request] = deque()

    def push(self, request: Request) -> None:
        if request.state is not RequestState.WAITING:
            raise RuntimeError("Only waiting requests can be enqueued")
        self._queue.append(request)

    def pop_batch(self, max_batch_size: int) -> list[Request]:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")

        batch: list[Request] = []
        while self._queue and len(batch) < max_batch_size:
            batch.append(self._queue.popleft())
        return batch

    def __bool__(self) -> bool:
        return bool(self._queue)


class StaticBatchScheduler:
    """Select the next fixed-size batch from the request queue."""

    def __init__(self, request_queue: RequestQueue, max_batch_size: int):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.request_queue = request_queue
        self.max_batch_size = max_batch_size

    def next_batch(self) -> list[Request]:
        return self.request_queue.pop_batch(self.max_batch_size)


@dataclass(slots=True)
class BatchState:
    """Decode-time tensors and cache for one admitted batch."""

    requests: list[Request]
    next_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    cache: object

    @property
    def finished(self) -> bool:
        return all(request.is_finished for request in self.requests)

    def replace_next_tokens(
        self, token_ids: Sequence[int], device: torch.device
    ) -> None:
        self.next_input_ids = torch.tensor(
            [[int(token_id)] for token_id in token_ids],
            device=device,
            dtype=torch.long,
        )


class ModelRunner:
    """Own the model forward passes for prefill and decode."""

    def __init__(
        self,
        model_name: str,
        pad_token_id: int,
        device: str,
        dtype: torch.dtype,
    ):
        self.device = torch.device(device)
        config = AutoConfig.from_pretrained(model_name)
        config.tie_word_embeddings = False
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            dtype=dtype,
        ).to(self.device)
        self.model.eval()
        self.pad_token_id = int(pad_token_id)

    def _build_prefill_inputs(
        self,
        requests: Sequence[Request],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        prompt_lens = [len(request.prompt_ids) for request in requests]
        max_prompt_len = max(prompt_lens)

        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for request in requests:
            pad_len = max_prompt_len - len(request.prompt_ids)
            input_rows.append([*request.prompt_ids, *([self.pad_token_id] * pad_len)])
            mask_rows.append([1] * len(request.prompt_ids) + [0] * pad_len)

        input_ids = torch.tensor(input_rows, device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(mask_rows, device=self.device, dtype=torch.long)
        return input_ids, attention_mask, prompt_lens

    @staticmethod
    def _greedy_select(logits: torch.Tensor) -> list[int]:
        return torch.argmax(logits, dim=-1).tolist()

    def prefill(self, requests: Sequence[Request]) -> BatchState:
        input_ids, attention_mask, prompt_lens = self._build_prefill_inputs(requests)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        first_token_ids: list[int] = []
        for row, _request in enumerate(requests):
            # The next token comes from the last real prompt position, not padding.
            last_prompt_index = prompt_lens[row] - 1
            next_token_logits = outputs.logits[row, last_prompt_index, :]
            first_token_ids.append(int(torch.argmax(next_token_logits).item()))

        next_input_ids = torch.tensor(
            [[token_id] for token_id in first_token_ids],
            device=self.device,
            dtype=torch.long,
        )
        return BatchState(
            requests=list(requests),
            next_input_ids=next_input_ids,
            attention_mask=attention_mask,
            cache=outputs.past_key_values,
        )

    def decode(self, batch: BatchState) -> list[int]:
        # [batch, 1]
        step_mask = torch.ones(
            (batch.attention_mask.shape[0], 1),
            device=self.device,
            dtype=batch.attention_mask.dtype,
        )
        # Each decode step extends the visible sequence by one generated token.
        # e.g. [1, 1, 0, 0] -> [1, 1, 0, 0, 1]
        batch.attention_mask = torch.cat([batch.attention_mask, step_mask], dim=1)
        outputs = self.model(
            input_ids=batch.next_input_ids,
            attention_mask=batch.attention_mask,
            past_key_values=batch.cache,
            use_cache=True,
            return_dict=True,
        )
        batch.cache = outputs.past_key_values
        return self._greedy_select(outputs.logits[:, -1, :])


class ServingSystem:
    """Minimal static-batching serving system built around a queue and decode loop."""

    def __init__(
        self,
        model_name: str,
        max_batch_size: int = 4,
        max_new_tokens: int = 64,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (
            torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.runner = ModelRunner(
            model_name=model_name,
            pad_token_id=int(self.tokenizer.pad_token_id),
            device=self.device,
            dtype=self.dtype,
        )
        self.request_queue = RequestQueue()
        self.scheduler = StaticBatchScheduler(
            request_queue=self.request_queue,
            max_batch_size=max_batch_size,
        )

    def submit(
        self,
        request_id: str,
        prompt_text: str,
        max_new_tokens: int | None = None,
        eos_token_id: int | None = None,
    ) -> Request:
        """Tokenize one prompt, build a request, and enqueue it."""
        request = Request(
            request_id=request_id,
            prompt_text=prompt_text,
            prompt_ids=tuple(
                self.tokenizer.encode(
                    prompt_text,
                    add_special_tokens=False,
                    verbose=False,
                )
            ),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            eos_token_id=self.eos_token_id if eos_token_id is None else eos_token_id,
        )
        self.request_queue.push(request)
        return request

    def run(self) -> list[Request]:
        completed: list[Request] = []
        while self.request_queue:
            batch = self.scheduler.next_batch()
            if not batch:
                break
            self._run_batch(batch)
            completed.extend(batch)
        return completed

    def _run_batch(self, requests: list[Request]) -> None:
        if not requests:
            return

        for request in requests:
            request.start()

        with torch.inference_mode():
            batch = self.runner.prefill(requests)
            for request, token_id in zip(
                batch.requests,
                batch.next_input_ids[:, 0].tolist(),
                strict=True,
            ):
                request.record_token(int(token_id))

            while not batch.finished:
                token_ids = self.runner.decode(batch)
                for request, token_id in zip(batch.requests, token_ids, strict=True):
                    if request.is_finished:
                        continue
                    request.record_token(int(token_id))

                # Feed the generated token back as the next decode input.
                batch.replace_next_tokens(
                    [request.output_ids[-1] for request in batch.requests],
                    device=self.runner.device,
                )


class MicroEngine(ServingSystem):
    """Backward-compatible wrapper for the original microengine entrypoint name."""
