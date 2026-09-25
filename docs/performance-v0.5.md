# Sustained benchmark timing

Update: the returned diagnostic report reached **16.13 tokens/s on external
power** in one measured 128-token run. See the [M5 result](m5-pro.md) for the
validated totals, configuration and limits. The investigation below describes
the evidence available before that report arrived.

The next user-reported M5 Pro results did not reproduce the single-step profile's
apparent throughput. Both used v0.3, aligned GEMV, parallel RMS, a 512-token
prefill, 128 generated steps and context capacity 8192:

| Metadata | Measured decode median | Reported GPU allocations |
| --- | ---: | ---: |
| FP32 | 7.44877 tokens/s | 18,055,610,368 bytes |
| BF16 | 8.03204 tokens/s | 16,374,677,504 bytes |

Raw user-provided records are retained in
[`m5-pro-v0.3-parallel-f32-user.json`](benchmarks/m5-pro-v0.3-parallel-f32-user.json)
and [`m5-pro-v0.3-parallel-bf16-user.json`](benchmarks/m5-pro-v0.3-parallel-bf16-user.json).
BF16 compacted 498 matrices, saving exactly 1,680,834,560 metadata bytes, and
improved the reported median by 7.83% relative to FP32 in these runs. It was
1.58% below the earlier 8.16096 aligned result. That earlier result came from a
separate run/version, so it is not a controlled estimate of the RMS change.
The available sustained measurements still do not establish 15 tokens/s.
The user confirmed that the two benchmarks ran sequentially, with no other
generation workload on the GPU. Concurrent model generation is therefore not
the explanation reported for this pair of runs.

## What the existing code explains

The short profiler and benchmark both call the same `Engine::forward`. Its wall
duration includes GPU encoding/wait, logits readback and the per-token autorelease
pool. Benchmark decoding additionally calls the greedy sampler. Temperature-zero
sampling has two linear scans of the logits, with no sorting or probability
allocation. A probe linked to the actual v0.4 release library on the local M1
measured a 1.059 ms median over 500 calls after 100 warmups at vocabulary 248,320.
This is a CPU microbenchmark, not target M5 evidence. The raw summary and library
hash are in [`m1-v0.4-sampler.json`](benchmarks/m1-v0.4-sampler.json).

The first 511 prefill tokens skip the output head/readback and have no sampling,
yet the target prefill is also around 7.5–9.3 tokens/s. This weakens an explanation
based solely on sampler or logits readback overhead. The reset happens before
the prefill timer, and progress logging happens after each measured run. There
is no per-token reset or log write. Attention uses actual history, not the full
8192-slot capacity. Its limited growth during the 128 decode steps does not
by itself identify the roughly 50–60 ms single-step/sustained gap.

There is not enough evidence to attribute that gap to GPU execution, host
scheduling, thermal behavior or competing workloads. The next measurement
must observe the actual sustained path rather than substitute a different GPU
scheduling pattern or extrapolate a short capture.

## The opt-in diagnostic

`bench --timing` reads the existing per-command timing and separately measures
sampling and forward wall time. It keeps the ordinary command graph: one encoder,
one command buffer and one completion wait per token. It does not use stage
counters or one-command-per-operation profiling.

All per-token storage is preallocated. JSON generation and percentile sorting
run after the timed phases. Host clocks, validation and scalar writes add some
observer overhead; the flag is diagnostic and not a new performance kernel.
The uninstrumented benchmark remains available for final acceptance.

Each run contains phase totals, distributions and 32-token block averages.
These distinguish a slow GPU interval from long completion waiting or time
outside the GPU submission/completion calls. GPU and wait overlap and must not
be added. Forward minus CPU encode/commit/wait is reported as `unclassified_host`:
it includes readback, autorelease, timestamp queries, validation and other host
work, not solely readback. Missing timestamps invalidate the corresponding
frame-derived phase/block aggregate instead of becoming zero-duration samples.
Forward and sampler measurements remain available with explicit coverage counts.

Warmup telemetry is retained under `warmup_run` and excluded from `runs` and the
median. Only the last prefill token computes logits; the report marks that fact.
At phase boundaries, read-only NSProcessInfo snapshots record Low Power Mode
and macOS thermal state. These are neither clock-frequency nor temperature
measurements. Their interpretation follows Apple's
[power and thermal state documentation](https://developer.apple.com/documentation/xcode/responding-to-power-notifications).
Unavailable APIs yield null fields, and no system setting is changed.

Use the same metadata, norm and attention options as the measurement being
diagnosed. For the latest BF16 result, keep attention serial and run:

```sh
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal bench --timing \
  --model models/Qwen3.8-27B-4bit --context 8192 \
  --prompt-tokens 512 --generate-tokens 128 --runs 1 > timing.json
```

This records one warmup and one measured run. It targets the unexplained
latency, rather than establishing a new three-run throughput median.
