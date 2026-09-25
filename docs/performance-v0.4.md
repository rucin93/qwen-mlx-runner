# M5 profile follow-up

The user supplied a successful v0.3 command-buffer profile of the trained local
checkpoint on an Apple M5 Pro. The unmodified values are retained in
[`benchmarks/m5-pro-v0.3-profile-user.json`](benchmarks/m5-pro-v0.3-profile-user.json).
This is user-reported target evidence, not a locally executed M5 run.

## What the target profile establishes

Both captures completed all 1,155 operations without an execution or timing
error. The ordinary, uninstrumented steps show:

| Measurement | Serial RMS | Parallel RMS |
| --- | ---: | ---: |
| Normal step wall time | 83.177 ms | 72.076 ms |
| Normal step GPU time | 82.097 ms | 70.816 ms |
| CPU encode time | 0.621 ms | 0.582 ms |
| History before the normal step | 513 | 516 |

The wall-time reduction is 13.35%, corresponding to a 1.154x ratio. These are
single adjacent-history steps, without the benchmark's additional greedy
sampling, not a replacement for a sustained 128-token, three-run benchmark.
The latest available full-model benchmark median remains 8.16096 tokens/s.

The separate instrumented graph measures 129 RMS operations at 14.568 ms total
in serial mode and 1.355 ms in parallel mode. This explains the direction of
the normal-step result. It does not make the instrumented and ordinary timings
interchangeable: the per-operation command backend changes scheduling and
adds overhead. In particular, unchanged matrix operations measured 79.386 ms
in the serial capture and 86.823 ms in the parallel capture, demonstrating
noise/variation in these diagnostic samples.

Matrix-vector multiplication dominates the remaining instrumented work. The
gate/up and down MLP projections total 52.280 ms in the serial capture. At equal
weight sizes, down took 246 microseconds per call and gate/up 285 microseconds;
the data does not support treating the down shape as an exceptional bottleneck.

The immediate controlled comparison is `QWEN_METAL_NORM=parallel` with aligned
matvec and FP32 metadata, followed by the same settings with BF16 metadata.
Both settings already exist in v0.3. They need full-model target benchmarking;
neither 15 tokens/s nor a sustained 10x improvement follows from this profile.

## Bounded attention follow-up

`attn_values` accounts for about 3 ms in each diagnostic capture at roughly
515 historical tokens. The original kernel uses one thread per output element,
with a sequential loop over the complete history. Splitting this reduction
across eight SIMD groups is a narrowly scoped candidate, especially for longer
histories. It cannot by itself explain or eliminate the matrix traffic cost.

The implementation retains FP32 scores, cache, accumulation and sigmoid gating,
and does not change cache layout or GQA head grouping. Enable it with
`QWEN_METAL_ATTN_VALUES=parallel`. History lengths below 128, head dimensions
below 32 and aliased outputs use the existing serial kernel. Reference mode
also stays serial. Each eligible group processes 32 adjacent channels, splits
history over eight SIMD groups, and reduces partial sums through threadgroup
memory. Bounds are checked for tail channels, which still reach the barrier.

## Local microbenchmark

The release example `attention_bench` uses identical FP32 inputs on one M1 GPU
and performs one production attention-value dispatch per command buffer. It
excludes five warmup iterations, measures 50 calls per case (10 at length 8192),
and reads outputs after the timed samples. Median actual GPU intervals were:

| History | Serial | Parallel | Ratio |
| ---: | ---: | ---: | ---: |
| 128 | 0.0920 ms | 0.0373 ms | 2.47x |
| 515 | 0.1292 ms | 0.0563 ms | 2.29x |
| 2048 | 0.8852 ms | 0.4394 ms | 2.01x |
| 8192 | 3.7034 ms | 1.9931 ms | 1.86x |

These are reused-buffer microbenchmarks on an M1, not a full-model or M5 result.
The modes run in serial-then-parallel order; GPU clocks and cache reuse can
influence the measurements. The largest absolute difference between the two
outputs was below 3.6e-7. Raw summaries, ranges and the benchmark binary hash are
in [`benchmarks/m1-v0.4-attention.json`](benchmarks/m1-v0.4-attention.json).

This candidate remains opt-in. The M5 profile suggests only a few-percent
opportunity at history 515, even if the kernel improves markedly. The larger
immediate opportunities are the already implemented parallel RMS and exact
BF16 metadata storage. Sustained target benchmarks determine the final choice.
