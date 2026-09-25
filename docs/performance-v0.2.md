# v0.2: target evidence, GPU profiling, and parallel normalization

The target is still at least 15 decode tokens/s on M5 Pro / 48 GB. This release
does not claim that result. It adds a measured-kernel optimization and the
diagnostics needed to explain the remaining gap on the actual machine.

## User-reported M5 Pro measurements

[Full model JSON](benchmarks/m5-pro-aligned-user.json) records the supplied
three-run benchmark at context 8192, 512 prompt tokens, and 128 generated steps:

- Median decode: **8.160957654 tokens/s**; individual runs 8.1722, 8.1610, 8.0617.
- Median prefill rate: **8.682439569 tokens/s**.
- GPU allocations: 18,055,610,368 bytes; load time: 22.567253 seconds.
- Reported mode: `aligned`; reported engine version: `0.1.0`.

Against the earlier supplied measured run of 1.452415808 decode tokens/s, this
is about 5.62×. The earlier full-run median was not supplied, so this is not a
comparison of two complete medians. Approximately another 1.84× is needed to
reach 15 tokens/s.

[The supplied matrix microbenchmarks](benchmarks/m5-pro-aligned-kernels-user.json)
show 0.247190 ms for 17408 × 5120 (225.36 GB/s effective payload) and
2.987590 ms for the 248320 × 5120 vocabulary projection (265.97 GB/s).
These repeated-buffer measurements do not establish sustained bandwidth or
throughput for the complete model with distinct layer weights.

All M5 numbers above were provided by the user. The commit and checkpoint
revision were not reported; no M5 execution occurred in this workspace.

## Changes

- Optional parallel RMS normalization uses eight SIMD groups (256 threads) instead of one,
  reducing the long serial loop. The FP32 sum is reduced through threadgroup
  memory with two unconditional barriers. Small or aliased buffers retain the
  original kernel. Enable it with `QWEN_METAL_NORM=parallel`; the old `serial`
  mode remains the default because a full-model gain is not yet established.
- The optional `QWEN_METAL_GEMV=stream` kernel assigns one Q4 group to each lane
  and reduces within four eight-lane row subgroups. It reduces repeated affine
  metadata arithmetic but increases input loads. It remains experimental and
  is not the default; local kernel gains do not establish a full-model gain.
- `profile` measures one normal command and a separately instrumented command.
  Normal inference still has one serial compute encoder per token. Profiling
  creates a compute encoder per operation, uses stage-boundary counters, and
  resolves samples through a blit into Shared memory after the compute passes.
- Whole-command GPU timestamps are read from `GPUStartTime` / `GPUEndTime`.
  CPU encoding, CPU commit, and completion wait are measured separately. The
  wait overlaps GPU execution and must not be added to GPU time.
- GPU counter ticks are calibrated using paired CPU/GPU timestamps surrounding
  the capture. Raw tick differences are not assumed to be nanoseconds. Invalid
  samples or clock calibration are errors, not zero-duration measurements.

The matrix, KV, and recurrent-state formats are unchanged. No model layers,
context slots, or output steps are skipped by the inference optimization.

## Local evidence and selection

[Calibrated M1 captures](benchmarks/m1-v0.2-profiles.json) use the synthetic
reused-weight graph at context 2048 and zeroed history 512. Two captures per RMS
mode produced these ranges:

| Measurement | Serial RMS | Parallel RMS |
| --- | ---: | ---: |
| Total RMS time in the instrumented token | 10.44–28.48 ms | 1.11–2.00 ms |
| GPU interval of the preceding normal token | 580.73–626.62 ms | 638.79–674.67 ms |

The isolated normalization improved, but the normal-token captures were slower
with parallel RMS. This active M1 desktop also showed substantial variability;
these data establish no full-model gain. For that reason serial RMS stays the
default, and the target profiler compares both settings before promotion.
Counter captures change encoder scheduling and do not provide a replacement
for a repeated model benchmark on the target.

## Target diagnosis

After building with `cargo build --release --locked`, capture the real model:

```sh
./target/release/qwen-metal profile \
  --model models/Qwen3.8-27B-4bit --context 8192 --history 512 \
  --compare-norm > profile.json
```

The command prefills 512 fixed tokens once. For each RMS mode, it runs a warmup
step, measures a normal step, and profiles the next step. It loads the checkpoint
only once. Steps have adjacent history lengths, reported in the JSON.
In each `captures` entry, inspect `normal_timing` for CPU/GPU separation, then `operations` for the most
expensive kernels and matrix shapes. Do not read the instrumented step's wall
time as normal token throughput: the extra encoders change GPU scheduling.

For the unchanged model benchmark and acceptance test:

```sh
./target/release/qwen-metal bench \
  --model models/Qwen3.8-27B-4bit \
  --context 8192 --prompt-tokens 512 --generate-tokens 128 --runs 3 > after.json
python3 scripts/compare_bench.py \
  docs/benchmarks/m5-pro-aligned-user.json after.json --require-tps 15
```

This comparison reports improvement against the 8.16 tokens/s result and checks
the absolute 15 tokens/s target. Keep checkpoint files, power
mode, context, and other running workloads unchanged.

## Correctness scope

The new RMS tests compare FP64 expectations for zero/nonzero inputs, odd and
large lengths, and in-place fallback. The packed matrix oracle covers Q4/Q8,
three group sizes, row/column tails and all selectable GEMV modes. Full-model
fixtures compare logits across tokens and resets, including a capture that
checks profiling leaves the numerical output unchanged. Clock aggregation
tests include a non-unit GPU/CPU clock ratio.

These are synthetic numerical tests on a real M1 GPU. They do not validate the
trained 27B model's text quality or prove a target-machine speedup for v0.2.
