from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from labs.baseline import Request


DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass(frozen=True, slots=True)
class MetricSummary:
    mean_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    model_name: str
    num_requests: int
    warmup_requests: int
    unique_prompts: int
    max_batch_size: int
    max_new_tokens: int
    device: str
    dtype: str
    elapsed_s: float
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    requests_per_s: float
    input_tokens_per_s: float
    output_tokens_per_s: float
    total_tokens_per_s: float
    ttft_ms: MetricSummary
    tpot_ms: MetricSummary
    e2e_ms: MetricSummary


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark the minimal baseline engine."
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model name.",
    )
    model_group.add_argument(
        "--device",
        default=None,
        help="Execution device. Defaults to cuda when available, else cpu.",
    )
    model_group.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default=None,
        help="Tensor dtype. Defaults to bfloat16 on cuda, else float32.",
    )

    workload_group = parser.add_argument_group("workload")
    workload_group.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text used when --prompt-file is not set.",
    )
    workload_group.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file with one prompt per line.",
    )
    workload_group.add_argument(
        "--numseqs",
        "--num-requests",
        dest="num_requests",
        type=int,
        default=8,
        help="Number of measured requests to submit.",
    )
    workload_group.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
        help="Warmup requests to run before the measured workload.",
    )
    workload_group.add_argument(
        "--max-batch-size",
        "--b",
        dest="max_batch_size",
        type=int,
        default=4,
        help="Static batch size.",
    )
    workload_group.add_argument(
        "--max-new-tokens",
        "--output-len",
        dest="max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per request.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of plain text.",
    )

    args = parser.parse_args()
    if args.num_requests < 1:
        raise ValueError("--num-requests must be >= 1")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if args.max_batch_size < 1:
        raise ValueError("--max-batch-size must be >= 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be >= 1")
    return args


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve a CLI dtype name to a torch dtype."""
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def load_prompts(args: argparse.Namespace) -> list[str]:
    """Load prompts from a file or fall back to one inline prompt."""
    if args.prompt_file is None:
        return [args.prompt]

    prompts = [
        line.strip()
        for line in args.prompt_file.read_text().splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")
    return prompts


def percentile(values: list[float], fraction: float) -> float | None:
    """Compute one percentile from a non-empty metric list."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_metric(values_s: list[float]) -> MetricSummary:
    """Aggregate latency values into mean and percentile summaries."""
    if not values_s:
        return MetricSummary(
            mean_ms=None,
            p50_ms=None,
            p95_ms=None,
            p99_ms=None,
        )

    values_ms = [value * 1000.0 for value in values_s]
    return MetricSummary(
        mean_ms=statistics.fmean(values_ms),
        p50_ms=percentile(values_ms, 0.50),
        p95_ms=percentile(values_ms, 0.95),
        p99_ms=percentile(values_ms, 0.99),
    )


def build_requests(
    engine,
    prompts: list[str],
    *,
    num_requests: int,
    request_prefix: str,
) -> list[Request]:
    """Submit requests by cycling through the available prompts."""
    return [
        engine.submit(
            f"{request_prefix}{index}",
            prompts[(index - 1) % len(prompts)],
        )
        for index in range(1, num_requests + 1)
    ]


def run_engine(engine) -> float:
    """Run the engine once and return the measured makespan."""
    started_at = time.perf_counter()
    for _output in engine.run():
        pass
    return time.perf_counter() - started_at


def summarize_requests(
    *,
    requests: list[Request],
    model_name: str,
    warmup_requests: int,
    unique_prompts: int,
    max_batch_size: int,
    max_new_tokens: int,
    device: str,
    dtype_name: str,
    elapsed_s: float,
) -> BenchmarkResult:
    """Aggregate per-request metrics into one result."""
    total_input_tokens = sum(len(request.prompt_ids) for request in requests)
    total_output_tokens = sum(len(request.output_ids) for request in requests)

    ttfts_s = [
        request.metrics.ttft_s
        for request in requests
        if request.metrics.ttft_s is not None
    ]
    tpots_s = [
        request.metrics.tpot_s
        for request in requests
        if request.metrics.tpot_s is not None
    ]
    e2e_s = [
        request.metrics.finished_at - request.metrics.submitted_at
        for request in requests
        if request.metrics.submitted_at is not None
        and request.metrics.finished_at is not None
    ]
    total_tokens = total_input_tokens + total_output_tokens

    return BenchmarkResult(
        model_name=model_name,
        num_requests=len(requests),
        warmup_requests=warmup_requests,
        unique_prompts=unique_prompts,
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype_name,
        elapsed_s=elapsed_s,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        requests_per_s=len(requests) / elapsed_s if elapsed_s > 0 else 0.0,
        input_tokens_per_s=(total_input_tokens / elapsed_s if elapsed_s > 0 else 0.0),
        output_tokens_per_s=(total_output_tokens / elapsed_s if elapsed_s > 0 else 0.0),
        total_tokens_per_s=total_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        ttft_ms=summarize_metric(ttfts_s),
        tpot_ms=summarize_metric(tpots_s),
        e2e_ms=summarize_metric(e2e_s),
    )


def run_benchmark(
    *,
    model_name: str,
    prompts: list[str],
    num_requests: int,
    warmup_requests: int,
    max_batch_size: int,
    max_new_tokens: int,
    device: str,
    dtype: torch.dtype,
    dtype_name: str,
) -> BenchmarkResult:
    """Run the benchmark workload and return a summary."""
    from labs.baseline import BaselineEngine

    engine = BaselineEngine(
        model_name=model_name,
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
    )

    if warmup_requests > 0:
        build_requests(
            engine,
            prompts,
            num_requests=warmup_requests,
            request_prefix="warmup-",
        )
        run_engine(engine)

    requests = build_requests(
        engine,
        prompts,
        num_requests=num_requests,
        request_prefix="req-",
    )
    elapsed_s = run_engine(engine)

    return summarize_requests(
        requests=requests,
        model_name=model_name,
        warmup_requests=warmup_requests,
        unique_prompts=len(set(prompts)),
        max_batch_size=max_batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype_name=dtype_name,
        elapsed_s=elapsed_s,
    )


def format_float(value: float | None, digits: int = 2) -> str:
    """Format an optional float for table output."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a compact aligned text table."""
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    separator_line = "  ".join("-" * width for width in widths)
    print(header_line)
    print(separator_line)
    for row in rows:
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def print_result(result: BenchmarkResult) -> None:
    """Print the result in a compact table format."""
    print(result.model_name)
    print(f"{result.device} / {result.dtype}")
    print()

    print_table(
        ["field", "value"],
        [
            ["requests", str(result.num_requests)],
            ["warmup", str(result.warmup_requests)],
            ["unique_prompts", str(result.unique_prompts)],
            ["batch", str(result.max_batch_size)],
            ["max_new_tokens", str(result.max_new_tokens)],
            ["elapsed_s", format_float(result.elapsed_s, digits=3)],
            ["requests_per_s", format_float(result.requests_per_s, digits=3)],
            ["input_tok_per_s", format_float(result.input_tokens_per_s, digits=3)],
            ["output_tok_per_s", format_float(result.output_tokens_per_s, digits=3)],
            ["total_tok_per_s", format_float(result.total_tokens_per_s, digits=3)],
            ["input_tokens", str(result.total_input_tokens)],
            ["output_tokens", str(result.total_output_tokens)],
            ["total_tokens", str(result.total_tokens)],
        ],
    )
    print()
    print_table(
        ["latency_ms", "mean", "p50", "p95", "p99"],
        [
            [
                "ttft",
                format_float(result.ttft_ms.mean_ms),
                format_float(result.ttft_ms.p50_ms),
                format_float(result.ttft_ms.p95_ms),
                format_float(result.ttft_ms.p99_ms),
            ],
            [
                "tpot",
                format_float(result.tpot_ms.mean_ms),
                format_float(result.tpot_ms.p50_ms),
                format_float(result.tpot_ms.p95_ms),
                format_float(result.tpot_ms.p99_ms),
            ],
            [
                "e2e",
                format_float(result.e2e_ms.mean_ms),
                format_float(result.e2e_ms.p50_ms),
                format_float(result.e2e_ms.p95_ms),
                format_float(result.e2e_ms.p99_ms),
            ],
        ],
    )


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_arguments()
    prompts = load_prompts(args)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    result = run_benchmark(
        model_name=args.model,
        prompts=prompts,
        num_requests=args.num_requests,
        warmup_requests=args.warmup_requests,
        max_batch_size=args.max_batch_size,
        max_new_tokens=args.max_new_tokens,
        device=device,
        dtype=resolve_dtype(dtype_name),
        dtype_name=dtype_name,
    )
    print_result(result)


if __name__ == "__main__":
    main()
