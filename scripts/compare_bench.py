#!/usr/bin/env python3
"""Compare two final `qwen-metal bench` JSON files (baseline, candidate).

Usage:
    python3 scripts/compare_bench.py baseline.json candidate.json
    python3 scripts/compare_bench.py baseline.json candidate.json --require-speedup 10

Only `model_fixed_token_benchmark` results are accepted. The command reads
files; it never runs the model or includes a warmup record in the medians.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


KIND = "model_fixed_token_benchmark"
WORKLOAD_FIELDS = (
    "model_path",
    "device",
    "context_capacity",
    "prompt_tokens",
    "generated_steps",
)


def positive_number(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{where} is too large") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{where} must be finite and positive")
    return number


def positive_integer(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{where} must be a positive integer")
    return value


def read_benchmark(path):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read final JSON: {exc}") from exc
    if not isinstance(result, dict) or result.get("kind") != KIND:
        raise ValueError(f"{path}: expected kind={KIND!r}; synthetic and kernel results are not model benchmarks")
    for field in ("model_path", "device"):
        if not isinstance(result.get(field), str) or not result[field]:
            raise ValueError(f"{path}: {field} must be a nonempty string")
    for field in ("context_capacity", "prompt_tokens", "generated_steps"):
        positive_integer(result.get(field), f"{path}: {field}")
    if result["prompt_tokens"] + result["generated_steps"] > result["context_capacity"]:
        raise ValueError(f"{path}: token workload exceeds context capacity")
    mode = result.get("kernel_mode")
    if mode is not None and (not isinstance(mode, str) or not mode):
        raise ValueError(f"{path}: kernel_mode must be a nonempty string when present")

    rows = result.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: runs must contain measured runs")
    run_numbers = set()
    decode_rates = []
    prefill_times = []
    for index, row in enumerate(rows):
        where = f"{path}: runs[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where} must be an object")
        run = positive_integer(row.get("run"), f"{where}.run")
        if run in run_numbers:
            raise ValueError(f"{where}: duplicate run number {run}")
        run_numbers.add(run)
        if row.get("warmup") is not False:
            raise ValueError(f"{where}: measured runs must have warmup=false")
        prefill = positive_number(row.get("prefill_seconds"), f"{where}.prefill_seconds")
        decode = positive_number(row.get("decode_seconds"), f"{where}.decode_seconds")
        rate = positive_number(row.get("decode_tokens_per_second"), f"{where}.decode_tokens_per_second")
        expected_rate = result["generated_steps"] / decode
        if not math.isclose(rate, expected_rate, rel_tol=1e-6):
            raise ValueError(f"{where}: decode rate disagrees with generated_steps/decode_seconds")
        if "prefill_tokens_per_second" in row:
            prefill_rate = positive_number(row["prefill_tokens_per_second"], f"{where}.prefill_tokens_per_second")
            if not math.isclose(prefill_rate, result["prompt_tokens"] / prefill, rel_tol=1e-6):
                raise ValueError(f"{where}: prefill rate disagrees with prompt_tokens/prefill_seconds")
        decode_rates.append(rate)
        prefill_times.append(prefill)
    if run_numbers != set(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: measured run numbers must be 1..{len(rows)} (warmup run 0 excluded)")
    median_decode = positive_number(
        result.get("median_decode_tokens_per_second"),
        f"{path}: median_decode_tokens_per_second",
    )
    if not math.isclose(median_decode, statistics.median(decode_rates), rel_tol=1e-6):
        raise ValueError(f"{path}: reported decode median disagrees with measured runs")
    return result, median_decode, statistics.median(prefill_times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="final JSON from the baseline bench")
    parser.add_argument("candidate", type=Path, help="final JSON from the candidate bench")
    parser.add_argument(
        "--require-speedup",
        type=float,
        metavar="RATIO",
        help="exit 1 when the decode speedup is below this positive ratio",
    )
    args = parser.parse_args()
    if args.require_speedup is not None and (
        not math.isfinite(args.require_speedup) or args.require_speedup <= 0
    ):
        parser.error("--require-speedup must be finite and positive")
    try:
        baseline, baseline_decode, baseline_prefill = read_benchmark(args.baseline)
        candidate, candidate_decode, candidate_prefill = read_benchmark(args.candidate)
        for field in WORKLOAD_FIELDS:
            if baseline[field] != candidate[field]:
                raise ValueError(
                    f"workload mismatch for {field}: baseline={baseline[field]!r}, "
                    f"candidate={candidate[field]!r}"
                )
        if len(baseline["runs"]) != len(candidate["runs"]):
            raise ValueError(
                "measured run count mismatch: "
                f"baseline={len(baseline['runs'])}, candidate={len(candidate['runs'])}"
            )
    except ValueError as exc:
        print(f"comparison rejected: {exc}", file=sys.stderr)
        return 2

    ratio = candidate_decode / baseline_decode
    print(f"Model: {baseline['model_path']} on {baseline['device']}")
    print(
        f"Workload: context {baseline['context_capacity']}, "
        f"prompt {baseline['prompt_tokens']}, decode {baseline['generated_steps']} steps"
    )
    print(
        "Kernel modes: "
        f"baseline {baseline.get('kernel_mode', 'unreported')}, "
        f"candidate {candidate.get('kernel_mode', 'unreported')}"
    )
    print(
        f"Median decode: {baseline_decode:.3f} -> {candidate_decode:.3f} tokens/s; "
        f"speedup {ratio:.3f}x"
    )
    print(
        f"Median prefill per measured run: {baseline_prefill:.3f} -> "
        f"{candidate_prefill:.3f} s (baseline/candidate {baseline_prefill / candidate_prefill:.3f}x)"
    )
    print(f">=10x decode target: {'PASS' if ratio >= 10 else 'FAIL'}")
    if args.require_speedup is not None:
        passed = ratio >= args.require_speedup
        print(f"Required {args.require_speedup:g}x decode speedup: {'PASS' if passed else 'FAIL'}")
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
