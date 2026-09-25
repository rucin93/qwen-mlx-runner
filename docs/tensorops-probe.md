# FP32 TensorOps experiment

Version 0.6.5 adds an isolated Rust/Metal matrix implementation in
[`block_tensorops_probe.rs`](../examples/block_tensorops_probe.rs) and
[`block_tensorops.metal`](../examples/shaders/block_tensorops.metal).
It is not selected by inference. The measured M5 Pro production choice remains
the scalar R2 MLP path documented in [the full-model comparison](m5-selective-r2.md).

The hypothesis is that Metal's matrix primitives can reduce target GPU work
more substantially than changing the scalar row schedule. The latest model
report still requires roughly 22% less target GPU time on its two slower Polish
prompts. A standalone primitive result cannot establish that improvement.

## Arithmetic and execution

The checkpoint remains packed affine Q4/group64 with exact BF16 scale/bias
metadata. Each shader reconstructs weights with FP32 `fma(q, scale, bias)` into
a cooperative tensor, then multiplies by the original FP32 activations. Both
inputs and the accumulator are FP32, `relaxed_precision=false`, and fast math
is disabled for this library. No inference framework or model conversion is
used. TensorOps is Apple's low-level Metal shader API.

The four candidates use one SIMD group, 16/32 output rows, 64/128 columns per
iteration, and eight batch columns. A small GPU dispatch copies FP32 activations
into an explicitly zero-padded buffer of `8 * round_up(cols, 128)` floats.
**This dispatch is included in every candidate GPU timing.** X uses unit
innermost stride and an explicit
transpose in the matmul descriptor; Y is written through guarded coordinates.
Weights are dequantized directly into registers without a full dense weight
buffer. Shape tails are tested separately from production-sized timing.

The initial inline tensor view relied on implicit boundary padding. An extended
test exposed an intermittent nonfinite result on M1 for rows=17, cols=192,
batch=1, M16/K128. Explicit padding replaces that assumption. Tests poison the
scratch buffer, verify every copied/zero-filled element bit for bit and retain
boundary guards. Earlier unpadded timing runs are not the final implementation.

Explicit weight reconstruction and matrix accumulation change rounding and
reduction order relative to the scalar affine dot product. The probe therefore
reports differing bits instead of demanding scalar bitwise equality. It checks
an independent FP64 sum using the original quantized values, scale/bias and
inputs, with a cancellation-safe tolerance of
`1e-6 + 1e-5 * sum(abs(FP64 products))`. Numerical error, trained-model greedy
agreement and performance are separate gates; this primitive check covers only
the first one.

Apple documents basic cooperative input support starting in macOS 26.3 and
supports inline tensors backed by ordinary Metal buffers. This probe requests
Metal Shading Language 4.0 and checks the OS version before compilation.
The installed `metal` crate predates that enum member, so the probe sends the
documented integer value through Objective-C without manufacturing an invalid
Rust enum. Normal engine startup never compiles this experimental library.
[Apple M5 GPU guidance](https://developer.apple.com/videos/play/tech-talks/111432/),
[Metal tensor operations](https://developer.apple.com/videos/play/wwdc2026/330/).

## Reproduce on M5

Use external power, Low Power Mode off, and no simultaneous generation. The
probe loads synthetic matrices, so it needs neither checkpoint directory nor
MTP adapter:

```sh
git pull --ff-only
cargo run --release --locked --example block_tensorops_probe -- --self-test > tensorops-check.json
cargo run --release --locked --example block_tensorops_probe -- --sweep > tensorops-m5.json
```

`--self-test` checks all outputs on 480 cases: four kernels, row counts 1/17/36,
column counts 64/192/512/5120/17408, batches 1/2/3/4, and two deterministic seeds.
The second seed spans a wider activation magnitude range. NaN poisoning,
nonzero input/output offsets, boundary guards and unchanged input values catch
incomplete writes and memory corruption.

`--sweep` compares each candidate against the production scalar implementation
on 17408×5120, 5120×17408 and 248320×5120, B3. The first two use R2/T64; the head
uses R4/T64. Pipeline thread limits and SIMD hints match production. It excludes
ten warmup pairs, records 24 alternating AB/BA GPU timestamp pairs, and checks
the FP64 oracle on 32 rows per batch before and after measurement. All output
values must be finite; input/output guards must remain intact. Invalid GPU
timestamps and numeric failures abort rather than produce a valid result.

The JSON records exact shader/harness hashes, OS/device, conditions, raw times,
oracle errors and bitwise differences. Repeated synthetic weights may benefit
from cache; these are kernel timings, never model tokens/s. Successful runtime
compilation also does not establish utilization of M5 GPU Neural Accelerators.
Only a winning target-device primitive followed by trained-model validation
would justify integrating a new path into inference.

## Local 0.6.5 evidence

The [480-case report](benchmarks/m1-v0.6.5-tensorops-selftest.json) checks 21,600
outputs. Three independent release processes produced identical reports
(SHA-256 `e28bb0e243d9814b533ebf9a10f93b247d9c8c67655973e1f061b33bcff840a5`).
All numerical, input-preservation, scratch-padding and boundary checks pass.
The maximum error divided by the FP64 sum of absolute products is
`4.83034e-7`; the maximum absolute error is 0.002690 on the wider-range fixture.
This is a precision check, not a trained-model output comparison.

The final [padded M1 sweep](benchmarks/m1-v0.6.5-tensorops-padded-sweep.json)
contains 12 configurations and 576 valid positive GPU durations. All oracle
checks pass before and after measurement. Every candidate median is slower
than its paired scalar baseline: ratios of scalar/candidate time range from
0.1176 to 0.2018, approximately **5–8.5 times slower**. The candidate wins only
one of 288 individual pairs; this does not support selecting TensorOps on M1.
All recorded power/thermal snapshots show Low Power Mode off and nominal state.
The final report SHA-256 is
`4b581de7f57197f1b759a05397a6679b3c576fae8ae69e9ad174040427c5b261`.

M5 performance is unmeasured. The prototype deliberately remains outside
inference while that question is unresolved. The release build, formatting
and full existing all-target suite also pass: **130 unique tests**, zero failed
or ignored, with actual Metal enabled on M1. The existing `block` dependency
future-compatibility warning remains.
