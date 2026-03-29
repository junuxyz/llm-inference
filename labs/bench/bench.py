from __future__ import annotations

import argparse
import random
import statistics
import time

import torch
from transformers import AutoTokenizer

from labs.baseline import SamplingConfig, ServingSystem

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
WORKLOAD = ("128x128", 128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the baseline engine on one fixed 128x128 workload."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=tuple(DTYPE_BY_NAME), default=None)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be >= 1")
    return args


def single_token_texts(tokenizer) -> tuple[list[str], list[str]]:
    cached = getattr(tokenizer, "_bench_single_token_texts", None)
    if cached is not None:
        return cached

    special_ids = {
        token_id
        for token_id in (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        )
        if token_id is not None
    }
    plain: list[str] = []
    spaced: list[str] = []
    vocab_size = max(int(tokenizer.vocab_size or len(tokenizer)), 1)

    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        text = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not text or not text.strip():
            continue
        if tokenizer.encode(text, add_special_tokens=False, verbose=False) != [
            token_id
        ]:
            continue
        if text[0].isspace():
            spaced.append(text)
        else:
            plain.append(text)

    if not plain:
        raise ValueError("Failed to find any stable single-token text pieces")
    cached = (plain, spaced or plain)
    tokenizer._bench_single_token_texts = cached
    return cached


def synthetic_prompt(tokenizer, prompt_tokens: int, seed: int) -> str:
    plain, spaced = single_token_texts(tokenizer)
    rng = random.Random(seed)
    first_candidates = rng.sample(plain, k=min(len(plain), 64))
    rest_candidates = rng.sample(spaced, k=min(len(spaced), 64))

    for first in first_candidates:
        if prompt_tokens == 1:
            return first
        for rest in rest_candidates:
            text = first + rest * (prompt_tokens - 1)
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if len(ids) == prompt_tokens:
                return text
    raise ValueError(f"Failed to build a prompt with exactly {prompt_tokens} tokens")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def summarize_ms(values_s: list[float]) -> list[float | None]:
    if not values_s:
        return [None, None, None, None]
    values_ms = [value * 1000.0 for value in values_s]
    return [
        statistics.fmean(values_ms),
        percentile(values_ms, 0.50),
        percentile(values_ms, 0.95),
        percentile(values_ms, 0.99),
    ]


def collect_metrics(requests: list[object]) -> dict[str, list[float | None]]:
    ttft_s: list[float] = []
    tpot_s: list[float] = []
    e2e_s: list[float] = []
    for request in requests:
        submitted_at = request.metrics.submitted_at
        first_token_at = request.metrics.first_token_at
        finished_at = request.metrics.finished_at
        if submitted_at is not None and first_token_at is not None:
            ttft_s.append(first_token_at - submitted_at)
        if (
            first_token_at is not None
            and finished_at is not None
            and len(request.output_ids) >= 2
        ):
            tpot_s.append(
                (finished_at - first_token_at) / (len(request.output_ids) - 1)
            )
        if submitted_at is not None and finished_at is not None:
            e2e_s.append(finished_at - submitted_at)
    return {
        "ttft": summarize_ms(ttft_s),
        "tpot": summarize_ms(tpot_s),
        "e2e": summarize_ms(e2e_s),
    }


def format_float(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def run_case(
    args: argparse.Namespace,
    tokenizer,
    device: str,
    dtype_name: str,
    dtype: torch.dtype,
    workload_name: str,
    prompt_tokens: int,
    output_tokens: int,
) -> None:
    total = args.num_requests + args.warmup_requests
    samples = [
        (
            synthetic_prompt(tokenizer, prompt_tokens, args.seed + index),
            SamplingConfig(max_new_tokens=output_tokens),
        )
        for index in range(total)
    ]
    warmup = samples[: args.warmup_requests]
    measured = samples[args.warmup_requests :]

    serve = ServingSystem(
        model_name=args.model,
        max_batch_size=args.max_batch_size,
        max_new_tokens=output_tokens,
        device=device,
        dtype=dtype,
    )

    for prompt_text, sampling in warmup:
        serve.submit(prompt_text=prompt_text, sampling=sampling)
    for _ in serve.run():
        pass

    requests = [
        serve.submit(prompt_text=prompt_text, sampling=sampling)
        for prompt_text, sampling in measured
    ]
    started_at = time.perf_counter()
    for _ in serve.run():
        pass
    elapsed_s = time.perf_counter() - started_at

    total_input_tokens = sum(len(request.prompt_ids) for request in requests)
    total_output_tokens = sum(len(request.output_ids) for request in requests)
    total_tokens = total_input_tokens + total_output_tokens
    metrics = collect_metrics(requests)

    print(args.model)
    print(f"{device} / {dtype_name}")
    print(f"workload: {workload_name} ({prompt_tokens}, {output_tokens})")
    print()
    print_table(
        ["field", "value"],
        [
            ["requests", str(args.num_requests)],
            ["warmup", str(args.warmup_requests)],
            ["batch", str(args.max_batch_size)],
            ["elapsed_s", format_float(elapsed_s, 3)],
            ["requests_per_s", format_float(args.num_requests / elapsed_s, 3)],
            ["input_tok_per_s", format_float(total_input_tokens / elapsed_s, 3)],
            ["output_tok_per_s", format_float(total_output_tokens / elapsed_s, 3)],
            ["total_tok_per_s", format_float(total_tokens / elapsed_s, 3)],
            ["input_tokens", str(total_input_tokens)],
            ["output_tokens", str(total_output_tokens)],
            ["total_tokens", str(total_tokens)],
        ],
    )
    print()
    print_table(
        ["latency_ms", "mean", "p50", "p95", "p99"],
        [
            ["ttft", *(format_float(value) for value in metrics["ttft"])],
            ["tpot", *(format_float(value) for value in metrics["tpot"])],
            ["e2e", *(format_float(value) for value in metrics["e2e"])],
        ],
    )


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    dtype = DTYPE_BY_NAME[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    workload_name, prompt_tokens, output_tokens = WORKLOAD
    run_case(
        args,
        tokenizer,
        device,
        dtype_name,
        dtype,
        workload_name,
        prompt_tokens,
        output_tokens,
    )


if __name__ == "__main__":
    main()
