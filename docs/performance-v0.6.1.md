# 0.6.1: reduce target verification work

The user's [0.6.0 M5 report](m5-mtp-v0.6.md) establishes 26.7659 sustained MTP
tokens/s and matching greedy traces. It spends 86.78% of decode time in target
model execution. This revision targets that cost. **No 0.6.1 full-model M5
throughput measurement is available yet; 32 tokens/s is not claimed.**

## Packed matrix blocks

For aligned Q4/group64 blocks of width 2 or 3, the kernel now unpacks each group
of packed weights once before applying it to the token vectors. It retains the
original FP32 dot-product and accumulation order. Widths 1 and 4 keep their
previous schedule; the shared-input variant regressed on width 4 during local
testing, so it was not enabled there.

The [same-input M1 benchmark](benchmarks/m1-block-shared-unpack-v0.6.1.json)
compares the actual production kernel with a frozen
copy of the 0.6.0 kernel, using 100 alternating paired observations and excluded
warmups. The retained candidate's B3 BF16 GPU ratios were approximately 1.13x
for the up projection, 1.02x for down and 1.13x for the vocabulary projection.
FP32-metadata measurements also improved. All tested outputs matched the frozen
kernel bit for bit and agreed with the independent FP64 oracle. These ratios
apply to isolated kernels and cannot be added to the recurrent ratio below or
multiplied directly into full-model tokens/s.

## Recurrent state across a block

For key dimensions up to 128, `delta_step_block` retains four FP32 state values
per SIMD lane across B1–B4 updates. It stores the state immediately before each
input as rollback prefix `0..B-1` and leaves the final result in the live buffer.
Full acceptance already uses live state, so no snapshot of prefix B is needed.
Convolution stays causal and its unused final snapshot is also omitted.
Key dimensions above 128 retain the existing sequential kernel.

At Qwen27B's 48 recurrent layers, live state is 144 MiB and convolution state is
5.625 MiB. The allocated snapshot buffers shrink from 748.125 to 598.5 MiB,
saving **149.625 MiB**. For B3, delta/state-copy dispatches drop from 336 to 48,
and convolution snapshot copies from 192 to 144. Total target dispatches fall
from 2471 to 2135. Model weights, state precision and equations are unchanged.

The [M1 pooled-state microbenchmark](benchmarks/m1-delta-snapshot-b3.json) uses all
48 layers' 144 MiB of nonzero state, 30 alternating AB/BA measured pairs and four
excluded warmup pairs. It measures:

| B3 recurrence and snapshots | Original | Fused |
| --- | ---: | ---: |
| Median GPU time | 32.526 ms | 15.928 ms |
| Median wall time | 33.270 ms | 16.485 ms |

The GPU ratio is **2.042x** for this primitive. Outputs, live state and every
retained snapshot agree bit for bit. Convolution and all target matrices are
outside this timing; it is not complete-generation or M5 throughput.

## Reproduction

```sh
cargo build --release --locked
```

On the target M5, connect external power and turn Low Power Mode off before
repeating the documented block-3 `mtp-bench --compare` command. Keep the
sampling settings, prompt list, maximum output count and context capacity the
same as the 0.6.0 report. The benchmark now warns before loading weights when
Low Power Mode is enabled. The OS settings are not changed by the program.

The block-4 follow-up ran with Low Power Mode on; its 11.25 tokens/s versus a
9.18 sequential baseline cannot be directly compared with block 3's normal
power result. Block 3 remains the default.

## Verification

The final combined 0.6.1 all-target suite passed **112 tests**, with zero failures
and zero ignored tests when Metal tests were enabled. The new recurrence checks
cover every width B1–B4, key dimensions 1/31/32/33/127/128, independent FP64
equations, exact original FP32 bit patterns, buffer guards, and invalid/aliasing
dispatches. A temporary nonzero K160 hybrid fixture exercises the fallback and
every rollback prefix through subsequent sequential continuation.

The matrix harness passes 32 independent-oracle cases spanning B1–B4,
FP32/BF16 metadata and widths 512/1536/5120/17408. It also compares every output
bit to its frozen baseline. Numerical engine tests, MTP acceptance/EOS/cancel
tests, benchmark accounting and HTTP unit tests remain in the full suite.

Independent source reviews found no blocking issues in either GPU change.
The BF16 block and MTP-manager suites also passed. The release binary reports
0.6.1 and its public `mtp-bench --compare` smoke test on tiny fixtures returns
matching greedy IDs with correct capture/count accounting. Formatting and
`git diff --check` passed.
These checks establish local correctness evidence and primitive speedups;
the new release still requires a controlled M5 full-model measurement.
