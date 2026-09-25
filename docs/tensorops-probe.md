# FP32 TensorOps experiment

Version 0.6.5 adds an isolated Rust/Metal matrix implementation in
[`block_tensorops_probe.rs`](../examples/block_tensorops_probe.rs) and
[`block_tensorops.metal`](../examples/shaders/block_tensorops.metal).
It is not selected by inference. The measured M5 Pro production choice remains
the scalar R2 MLP path documented in [the full-model comparison](m5-selective-r2.md).

**Decision after the user's M5 Pro run:** reject this FP32 B3 implementation
for inference. All 288 paired samples are slower than the existing scalar
kernels; median slowdowns span 4.21–8.10×. The 480-case numerical check passes.
This closes the four tested configurations, not every possible TensorOps
algorithm, precision or prefill workload. See the audited M5 evidence below.

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

## Reproduce the rejected M5 experiment

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

The 0.6.5 release build, formatting and full existing all-target suite also
passed: **130 unique tests**, zero failed or ignored, with actual Metal enabled
on M1. The existing `block` dependency future-compatibility warning remains.
The later M5 report below changes the performance decision, not the runtime.

## User M5 Pro result

The user supplied both reports from Apple M5 Pro, macOS 26.6.2, engine 0.6.5:

- [Numerical checks](benchmarks/m5-pro-v0.6.5-tensorops-selftest-user.json),
  SHA-256 `eb88857f540919b59bd1576eb72bdd0fc7544d2259ffb42420b253b1b6de932b`.
- [Paired GPU sweep](benchmarks/m5-pro-v0.6.5-tensorops-sweep-user.json),
  SHA-256 `84678dc4c3294676bfc29a3b8a9565b5d09c0aec10fc1f8424927dd3ce2dcdba`.

Both record exactly the checked-in harness and shader hashes. The numerical
report covers the complete 480-case Cartesian product with no duplicates,
checking 21,600 output values. Its maximum absolute error is 0.00268975 on the
wider-range input fixture; maximum error divided by the FP64 sum of absolute
products is 4.83034e-7. These checks meet the recorded tolerance and match the
earlier M1 per-case oracle summaries. They do not establish trained-model
greedy agreement, and the output is not bitwise identical to the scalar path.

The performance report has all 12 kernel/shape combinations, each with 24
alternating pairs after ten excluded warmup pairs: **576 positive finite GPU
durations**. Recomputed medians, ratios and pair-win counts match the report.
All 24 before/after snapshots show Low Power Mode off and nominal thermal state.
Activation padding is included in candidate timing.

| Candidate | Gate/up slowdown | Down slowdown | Vocabulary slowdown |
| --- | ---: | ---: | ---: |
| M16/K64 | 4.8022× | 4.2089× | 4.9435× |
| M16/K128 | 6.0441× | 5.2678× | 6.4351× |
| M32/K64 | 6.6737× | 7.1730× | 6.7478× |
| M32/K128 | 7.9671× | 7.6804× | 8.1009× |

Each entry divides that candidate's median time by its contemporaneous scalar
control. M16/K64 is the best tested candidate on all three shapes, but remains
substantially slower:

| Matrix | Scalar median | M16/K64 median | Scalar route |
| --- | ---: | ---: | --- |
| 17408 × 5120 | 0.257687 ms | 1.237479 ms | R2/T64 |
| 5120 × 17408 | 0.228417 ms | 0.961375 ms | R2/T64 |
| 248320 × 5120 | 2.745354 ms | 13.571562 ms | R4/T64 |

The first gate/up control falls from 0.3220 to 0.2247 ms across its pairs;
candidate time falls too. This makes cross-configuration baselines inappropriate.
Even with that drift, **every paired candidate sample is slower**. The rejection
therefore does not depend on choosing a different configuration's faster control.

FP64 checks also pass before and after every timed configuration. Largest sampled
candidate absolute error is 5.02615e-5, versus 3.17258e-6 for the scalar controls.
Different output bits are expected from the changed arithmetic and are explicitly
reported; the prototype has neither a precision advantage nor a speed advantage
in this experiment.

Do not integrate this implementation or request a full-model run for it. Keep R2
for the measured M5 Pro route. The data do not measure accelerator utilization
or separate dequantization, matrix computation and padding costs; they cannot
identify one of those as the cause, or rule out different TensorOps algorithms.
The most recent full-model result remains 28.69 sustained tokens/s overall,
with about 26 tokens/s on the two slowest prompts. The 32 tokens/s mixed-use
goal is still unconfirmed.
