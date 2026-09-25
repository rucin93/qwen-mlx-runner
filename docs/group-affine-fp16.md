# Group-affine mixed-precision experiment

## Objective and boundaries

The user requested an algorithm that processes tokens more efficiently than
the FP32 kernels after merging PR #1. This experiment retains the existing
Qwen3.8-27B Q4/group64 checkpoint and the independent Rust/Metal runtime.
It changes arithmetic, so bitwise equality and unchanged trained-model output
must not be assumed. It does not substitute Bonsai or requantize the model.

The production comparison is the measured R2 kernel on B3 MLP shapes and R4
elsewhere, not the rejected FP32 TensorOps prototype. B3 and B4 are measured
separately: a fixed eight-column matrix tile may amortize verification better
at B4, but accepted output per round and full-model cost remain unmeasured.

## Algorithm

For a group of 64 weights, the stored affine weight is `w = s*q + b`, with
`q` an integer in `[0,15]`. Center the integer without losing information:

```
t = q - 8
d = b + 8*s
sum(w*x) = s * sum(t*x) + d * sum(x)
```

Every centered integer `t` is exactly representable in FP16. Keep `s`, `b`,
the original activation sum and output accumulation in FP32. For each token
and activation group, choose a power-of-two scale `alpha` from the largest
absolute input. Zero groups use one; subnormal-only groups clamp the scale
to the smallest normal FP32 value. Compute:

```
v = x / alpha
high = FP16(v)
low  = FP16((v - FP32(high)) * 2048)
```

The single mode uses `high`. The compensated control represents `v` as
`FP32(high) + FP32(low)/2048`, performing a second matrix product for the
residual. Scaling the residual by 2048 avoids unnecessarily losing it in the
FP16 subnormal range. Compensation adds work; it is a precision option, not
an assumed speedup.

Within each group, multiply centered weights by FP16 activations using FP32
matrix accumulation. Combine the high/low products before applying `s*alpha`.
Apply the affine offset to the **original FP32 activation sum**, not the
rounded activation reconstruction. Then accumulate group contributions in
FP32. Unlike directly reconstructing every weight in FP16, this preserves
the original scale/bias representation and avoids rounding each weight.

The preparation kernel runs once per matrix input and writes every high/low
and metadata value, including padded token positions. Its full GPU time is
included in comparisons. Weights remain packed outside the matrix primitive;
there is no dense full-weight allocation.

## Packing precision into unused token positions

The B3/B4 workload occupies at most four positions in an eight-position matrix
tile. The packed compensated variant places `high` in positions 0–3 and the
scaled `low` residual in positions 4–7. A single matrix operation computes
both components. Their FP32 sums remain separate across groups, with the
affine offset added only to the high component. A final 512-byte threadgroup
tile combines the two components before writing each token's output.

This removes the second matrix call from the compensated mode. It still adds
preparation, storage and a final local synchronization, so a speedup must be
measured. Summing the two components after the group loop also changes rounding
relative to the two-call control. All modes use the same independent oracle.

The native Q4 variants additionally expose the weight matrix directly as a
signed 4-bit Metal tensor. XOR each stored word with `0x88888888` once during
loading: interpreting the resulting nibbles as signed integers represents
`q-8` exactly. This is a reversible internal encoding change, not another
quantization of the checkpoint. These variants avoid explicitly expanding
weights into cooperative FP16 storage. Their preparation and matrix time are
measured together; the one-time encoding and padding cost is outside GPU decode
timing and would belong to model loading if integrated.

Centering with an original-input affine correction is an approximation when
the activations are rounded. For example, `q=0,b=0` has zero original weights,
but a single-half approximation can produce a nonzero cancellation residual.
The test harness records this case separately. Passing the algorithm oracle
does not mean passing an application quality test or preserving greedy tokens.

## Implementation and acceptance plan

1. Implement isolated Metal kernels and a Rust harness without changing the
   inference dispatch or the historical FP32 probe.
2. Check all GPU preparation values against independent host FP16 conversions,
   independently check original group sums, and guard every scratch buffer.
3. Compare GPU output with two FP64 calculations: the original affine model
   and the explicitly approximated algorithm. Record input-reconstruction
   error separately from matrix arithmetic error. Include zeros, exact affine
   cancellation, values exceeding the FP16 range and small input magnitudes.
4. Measure against the existing scalar kernels using paired alternating
   GPU timings, with compilation/loading/warmup excluded and preparation
   included. Cover both MLP shapes and the vocabulary matrix at B3 and B4.
5. Preserve raw data and source fingerprints. Abort timing on an implementation
   oracle failure. Record approximation failures as negative evidence even if
   a candidate is timed; those failures exclude it from production consideration.
   A local M1 win does not establish an M5 win.
6. Only a target-device winner justifies inference integration and trained-model
   quality/greedy comparison. No 32 tokens/s claim follows from a matrix test.

This is a bounded kernel experiment. FP32 remains the baseline for reductions,
recurrent state and output. No new inference dependency is introduced.

## Running the experiment

The complete probe requires macOS 26.4 or later for the native signed-INT4
tensor operand. The existing inference binary has no new OS requirement.
The host compiles embedded Metal 4 source at runtime, so the separately
downloaded offline Metal compiler is not needed.

```sh
cargo build --release --locked --example block_affine_fp16_probe
./target/release/examples/block_affine_fp16_probe --self-test > affine-check.json
./target/release/examples/block_affine_fp16_probe --sweep > affine-sweep.json
```

Run on external power, with Low Power Mode disabled and no other GPU generation.
The fixed sweep includes B3 and B4 at `17408x5120`, `5120x17408` and
`248320x5120`. It compares five candidates, with ten excluded warmup pairs and
24 measured AB/BA pairs per configuration. Native INT4 weight repacking is
performed once before timing; activation preparation is included in every
candidate interval. Repeated synthetic weights can benefit from cache.

The self-test checks implementation arithmetic, preparation bits and guards.
It also **records approximation failures without aborting**, including the
zero-weight cancellation diagnostic. An exit code of zero therefore does not
mean that `all_original_error_gates_passed` is true. Inspect that field and the
individual `oracle` records. The timing sweep uses normal random inputs only;
it cannot replace the broader self-test or a trained-model comparison.

The use of packed integer tensor operands follows Apple's public
[Metal guidance for M5](https://developer.apple.com/videos/play/tech-talks/111432/).
Availability of an API is not evidence that this implementation wins on a
specific chip or uses a particular hardware execution unit.

## Rejected tile during development

An initial 32-row FP16 cooperative-input variant compiled, but its isolated
M1 test returned a nonfinite/unwritten output at `1x64`, B1, with Metal API and
GPU validation enabled. An earlier combined run also ended in a Metal GPU
recovery error. The root cause is unresolved; this is not evidence of a general
Metal or M5 limitation. Both 32-row entry points were removed from the executable
probe before timing. The five retained candidates all use 16-row tiles and must
pass their own tests; no numerical threshold was relaxed to admit the failed tile.
The [validation error](benchmarks/m1-v0.6.6-affine-m32-rejected.txt) is retained.

## Local M1 evidence, 2026-09-25

The release-build [self-test](benchmarks/m1-v0.6.6-affine-selftest.json) covers
500 cases and 18,000 outputs. All preparation-bit, guard and implementation
oracle checks pass. The three compensated variants each pass all 100 original
approximation checks. Each single-FP16 variant fails all 20 `q=0,bias=0`
absolute-error checks: its maximum error there is `0.003643900156`, versus the
predeclared `1e-6` limit. The native packed compensated maximum for that fixture
is `8.188653737e-7`. Its maximum error divided by the sum of absolute affine
components across all patterns is `4.583167954e-8`; this is **not** relative
error in the final output or a trained-model quality metric.

The [initial sweep](benchmarks/m1-v0.6.6-affine-sweep.json) contains 30
configurations, 720 AB/BA pairs and 1,440 positive finite GPU durations. The
[native packed repeat](benchmarks/m1-v0.6.6-affine-native-packed-repeat.json)
adds 144 pairs. All recorded condition snapshots report nominal thermals and
Low Power Mode off. Nevertheless, timings vary substantially, including
large baseline outliers. A bounded repeat was therefore required before
interpreting the initial B4 gains.

For the **native INT4 + packed compensated FP16** candidate, speedup is scalar
median GPU time divided by candidate median GPU time. Values above one favor
the candidate. Preparation is included.

| Batch | Matrix rows × columns | Initial speedup | Repeat speedup | Repeat faster pairs |
|---|---:|---:|---:|---:|
| 3 | 17408 × 5120 | 0.822× | 0.691× | 8/24 |
| 3 | 5120 × 17408 | 0.683× | 0.527× | 3/24 |
| 3 | 248320 × 5120 | 0.791× | 0.691× | 0/24 |
| 4 | 17408 × 5120 | 1.495× | 0.783× | 6/24 |
| 4 | 5120 × 17408 | 1.271× | 0.551× | 5/24 |
| 4 | 248320 × 5120 | 1.111× | 0.984× | 9/24 |

There is **no demonstrated stable M1 speedup for the compensated candidate**.
The expanded-FP16 candidates lose every configuration's median comparison;
native single-FP16 has some faster medians but fails the cancellation error
criterion above. Neither observation justifies replacing the production path.
Both complete timing reports are retained, including the unfavorable repeat.
There is no M5 result for this algorithm yet, and no full-model integration or
32 tokens/s claim.

All source fingerprints in these three JSON files match the published probe,
three shaders and production scalar control. The existing Rust test suite also
passes: 93 ordinary tests plus 43 separately selected GPU tests, 136 distinct
tests with no failures. The standalone self-test is additional to those tests.

The suite was executed in two non-overlapping selections (one GPU regression
test also launches a child process with the same test name):

```sh
cargo test --locked --all-targets -- --test-threads=1
cargo test --locked --all-targets -- --ignored --test-threads=1
cargo fmt --all -- --check
```

For a focused M5 measurement, use the same candidate in both invocations:

```sh
cargo build --release --locked --example block_affine_fp16_probe
./target/release/examples/block_affine_fp16_probe --self-test \
  --kernel affine_native_q4_m16_packed > affine-check.json
./target/release/examples/block_affine_fp16_probe --sweep \
  --kernel affine_native_q4_m16_packed > affine-m5.json
```

This requires no checkpoint download or model loading. It checks 100 cases
and measures all six B3/B4 matrix shapes against R2/R4. A target-device win and
passing numerical checks would justify the next step: opt-in integration and
real prompt, state and output-quality comparisons.
