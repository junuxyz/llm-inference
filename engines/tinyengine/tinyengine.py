from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RequestState(StrEnum):
    """
    Lifecycle state of a generation request.
    """
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """
    Minimal generation controls for tinyengine.
    """
    max_new_tokens: int = 64
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")


@dataclass(slots=True)
class RequestMetrics:
    """
    Per-request timestamps used by the benchmark harness.
    """
    submitted_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    token_timestamps: list[float] = field(default_factory=list)

    @property
    def first_token_at(self) -> float | None:
        return self.token_timestamps[0] if self.token_timestamps else None

    @property
    def ttft_s(self) -> float | None:
        if self.submitted_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.submitted_at

    @property
    def tpot_s(self) -> float | None:
        if len(self.token_timestamps) < 2:
            return None
        first_token_at = self.token_timestamps[0]
        last_token_at = self.token_timestamps[-1]
        return (last_token_at - first_token_at) / (len(self.token_timestamps) - 1)


@dataclass(slots=True)
class Request:
    """
    Per-request state tracked by the inference engine.
    """
    request_id: str
    prompt_text: str
    prompt_ids: tuple[int, ...]
    sampling: SamplingConfig
    output_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    finish_reason: str | None = None
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_text:
            raise ValueError("prompt_text must not be empty")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")

    @property
    def is_finished(self) -> bool:
        return self.state is RequestState.FINISHED

    @property
    def generated_len(self) -> int:
        return len(self.output_ids)

    @property
    def last_token_id(self) -> int:
        if not self.output_ids:
            raise RuntimeError("Request has not generated any output token yet")
        return self.output_ids[-1]

    def mark_submitted(self, now: float) -> None:
        if self.metrics.submitted_at is None:
            self.metrics.submitted_at = now

    def start(self, now: float) -> None:
        if self.state is not RequestState.WAITING:
            raise RuntimeError(f"Cannot start request in state={self.state}")
        self.state = RequestState.RUNNING
        if self.metrics.started_at is None:
            self.metrics.started_at = now

    def finish(self, reason: str, now: float) -> None:
        self.state = RequestState.FINISHED
        self.finish_reason = reason
        self.metrics.finished_at = now

    def fail(self, message: str, now: float) -> None:
        self.state = RequestState.FAILED
        self.finish_reason = message
        self.metrics.finished_at = now

    def record_token(self, token_id: int, now: float) -> None:
        if self.state is RequestState.FINISHED:
            raise RuntimeError("Cannot append token to a finished request")
        if self.state is RequestState.FAILED:
            raise RuntimeError("Cannot append token to a failed request")

        token_id = int(token_id)
        self.output_ids.append(token_id)
        self.metrics.token_timestamps.append(now)

        if self.generated_len >= self.sampling.max_new_tokens:
            self.finish("max_new_tokens", now)
            return
        if self.sampling.eos_token_id is not None and token_id == self.sampling.eos_token_id:
            self.finish("eos", now)

    def full_token_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.output_ids]


class TokenizerCodec:
    """
    Shared tokenizer wrapper used by both the engine and the benchmark.
    """
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> int | None:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            return int(eos_token_id)
        return 0

    @property
    def max_sequence_length(self) -> int | None:
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if not isinstance(model_max_length, int):
            return None
        if model_max_length <= 0 or model_max_length >= 1_000_000:
            return None
        return model_max_length

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(text, add_special_tokens=False, verbose=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)


class RequestQueue:
    """
    FIFO queue of incoming generation requests.
    """
    def __init__(self, clock: Callable[[], float]):
        self._queue: deque[Request] = deque()
        self._clock = clock

    def push(self, request: Request) -> None:
        if request.state is not RequestState.WAITING:
            raise RuntimeError("Only waiting requests can be enqueued")
        request.mark_submitted(self._clock())
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
    """
    1) admit up to max_batch_size requests
    2) run that batch to completion
    3) admit the next batch
    """
    def __init__(self, request_queue: RequestQueue, max_batch_size: int = 4):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.request_queue = request_queue
        self.max_batch_size = max_batch_size

    def has_work(self) -> bool:
        return bool(self.request_queue)

    def next_batch(self) -> list[Request]:
        return self.request_queue.pop_batch(self.max_batch_size)


@dataclass(slots=True)
class BatchState:
    """
    State that stays alive across decode steps for one admitted batch.
    """
    requests: list[Request]
    next_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    cache: object

    @property
    def finished(self) -> bool:
        return all(request.is_finished for request in self.requests)

    def replace_next_tokens(self, token_ids: Sequence[int], device: torch.device) -> None:
        self.next_input_ids = torch.tensor(
            [[int(token_id)] for token_id in token_ids],
            device=device,
            dtype=torch.long,
        )


class ModelRunner(Protocol):
    """
    Tiny interface around the model forward passes.
    """
    device: torch.device

    def prefill(self, requests: Sequence[Request]) -> BatchState:
        ...

    def decode(self, batch: BatchState) -> list[int]:
        ...


class HFModelRunner:
    """
    HuggingFace-based ModelRunner for batched prefill and static decode.
    """
    def __init__(
        self,
        model_name: str,
        pad_token_id: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
        # parallel forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        first_token_ids: list[int] = []
        for row, _request in enumerate(requests):
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
        step_mask = torch.ones(
            # [B, 1]
            (batch.attention_mask.shape[0], 1),
            device=self.device,
            dtype=batch.attention_mask.dtype,
        )
        # e.g. [1,1,0,0] -> [1,1,0,0,1]
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


@dataclass(frozen=True, slots=True)
class TokenEvent:
    """
    One generated token emitted by the engine.
    """
    request: Request
    token_id: int


class TinyEngine:
    """
    Clean tiny engine with explicit seams:
    - RequestQueue owns admission order
    - StaticBatchScheduler decides which requests enter a batch
    - ModelRunner owns model forward passes
    - BatchState owns cache + tensors across decode steps
    """
    def __init__(
        self,
        codec: TokenizerCodec,
        runner: ModelRunner,
        max_batch_size: int = 4,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.codec = codec
        self.runner = runner
        self.clock = clock
        self.request_queue = RequestQueue(clock=clock)
        self.scheduler = StaticBatchScheduler(
            request_queue=self.request_queue,
            max_batch_size=max_batch_size,
        )

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        max_batch_size: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        clock: Callable[[], float] = time.perf_counter,
    ) -> TinyEngine:
        codec = TokenizerCodec(model_name)
        runner = HFModelRunner(
            model_name=model_name,
            pad_token_id=codec.pad_token_id,
            device=device,
            dtype=dtype,
        )
        return cls(
            codec=codec,
            runner=runner,
            max_batch_size=max_batch_size,
            clock=clock,
        )

    @property
    def eos_token_id(self) -> int | None:
        return self.codec.eos_token_id

    def submit(
        self,
        request_id: str,
        prompt_text: str,
        sampling: SamplingConfig | None = None,
    ) -> Request:
        sampling = sampling or SamplingConfig()
        if sampling.eos_token_id is None:
            sampling = replace(sampling, eos_token_id=self.eos_token_id)

        request = Request(
            request_id=request_id,
            prompt_text=prompt_text,
            prompt_ids=self.codec.encode(prompt_text),
            sampling=sampling,
        )
        self.request_queue.push(request)
        return request

    def decode_output(self, request: Request) -> str:
        return self.codec.decode(request.output_ids)

    def decode_full(self, request: Request) -> str:
        return self.codec.decode(request.full_token_ids())

    def run(self) -> Iterator[TokenEvent]:
        while self.scheduler.has_work():
            batch_requests = self.scheduler.next_batch()
            if not batch_requests:
                break
            yield from self._run_batch(batch_requests)

    def _run_batch(self, requests: list[Request]) -> Iterator[TokenEvent]:
        if not requests:
            return

        for request in requests:
            request.start(self.clock())

        with torch.inference_mode():
            batch = self.runner.prefill(requests)
            first_step_time = self.clock()
            for request, token_id in zip(
                batch.requests,
                batch.next_input_ids[:, 0].tolist(),
                strict=True,
            ):
                request.record_token(int(token_id), now=first_step_time)
                yield TokenEvent(request=request, token_id=int(token_id))

            while not batch.finished:
                token_ids = self.runner.decode(batch)
                step_time = self.clock()

                for request, token_id in zip(batch.requests, token_ids, strict=True):
                    if request.is_finished:
                        continue
                    request.record_token(int(token_id), now=step_time)
                    yield TokenEvent(request=request, token_id=int(token_id))

                batch.replace_next_tokens(
                    [request.last_token_id for request in batch.requests],
                    device=self.runner.device,
                )


def main() -> None:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    engine = TinyEngine.from_model_name(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_batch_size=2,
    )
    requests = [
        engine.submit("r1", "I am", SamplingConfig(20)),
        engine.submit("r2", "I like", SamplingConfig(max_new_tokens=20)),
        engine.submit("r3", "I love", SamplingConfig(max_new_tokens=20)),
        engine.submit("r4", "I might", SamplingConfig(max_new_tokens=20)),
    ]

    for event in engine.run():
        print(
            f"{event.request.request_id}: {engine.decode_output(event.request)}",
            flush=True,
        )

    print("\nFinal outputs:")
    for request in requests:
        print(f"{request.request_id}: {engine.decode_full(request)}")


if __name__ == "__main__":
    main()


BaselineEngine = TinyEngine
