# M5 block-kernel comparison: shared unpacking causes the regression

Follow-up: the [M5 row-tile sweep](m5-selective-r2.md) rejects the R8 candidate
described below and supports testing R2 only for the two measured MLP shapes.
Version 0.6.4 adds that opt-in model comparison. The M1 measurements below
remain local primitive evidence and are not M5 speedups.

The user's [complete 0.6.2 report](benchmarks/m5-pro-v0.6.2-kernel-comparison-user.json)
compares four configurations in one process with one loaded target and MTP
adapter. Its SHA-256 is
`43d02247c034c9f2783ecfd6751dc8b681a7f0662207d3057d465217478dfa17`.
There are two prompts, four measured runs per prompt/configuration, and one
excluded warmup each: 40 captures in total. **This is not the full five-prompt
mixed workload and does not establish the 32 tokens/s goal.**

| Matrix / DeltaNet | Weighted sustained tokens/s | Polish median | Code median | Target GPU ms/round |
| --- | ---: | ---: | ---: | ---: |
| legacy / sequential | 27.9053 | 24.9469 | 32.0810 | 77.0721 |
| shared / sequential | 24.1545 | 21.4039 | 27.7586 | 91.0894 |
| **legacy / batched** | **28.5274** | **25.5858** | **32.9792** | **75.1125** |
| shared / batched | 24.6332 | 21.8329 | 28.2985 | 89.0721 |

Rates exclude the first token, which comes from prefill. Each configuration
produces 1,016 sustained tokens over eight measured generations. For historical
comparison on these same two prompts, 0.6.0 achieved 27.7548 and 0.6.1 achieved
24.6702. The prior 26.7659 figure covers five prompts and must not be directly
compared with this narrower workload.

## What the controlled comparison establishes

Shared unpacking is slower in **all 16 matched measured comparisons**, covering
both DeltaNet choices. It adds 18.19% GPU time with sequential DeltaNet and
18.58% with batched DeltaNet; CPU encoding changes by only 0.24–0.28%. This
identifies the matrix variant and GPU execution as the regression source.
It does not identify the microarchitectural cause. Higher register pressure is
a hypothesis, not a measured hardware-counter finding.

Batched DeltaNet improves legacy-matrix weighted throughput by **2.23%**,
reduces target time by 2.55%, and wins seven of eight decode wall-time pairs
and all eight target-time pairs. The final code run is effectively tied in
wall time (batched is 0.0196% slower). The improvement is small; it is not a
universal 2.23% guarantee across workloads or machines.

Version **0.6.3 makes legacy matrix + batched DeltaNet the default**. The
sequential recurrence fallback remains available, including for key dimensions
above 128. Both flags can still be set explicitly:

```sh
QWEN_METAL_BLOCK_MATMUL=legacy QWEN_METAL_BLOCK_DELTA=batched
```

The underlying winning kernels are unchanged from those in the user's 0.6.2
comparison. This default selection introduces no weight/precision changes.

## Integrity and conditions

- Execution order matches the declared balanced schedule. Each configuration
  occupies each timing position once per prompt over the four measured runs.
- All token IDs, texts, finish reasons, proposal/acceptance counts and round
  widths match across configurations, and match the corresponding prompts in
  the archived 0.6.0 and 0.6.1 reports.
- All 1,616 measured target command buffers have GPU timings; none are missing.
- Low Power Mode is false in all 64 measured condition snapshots.
- The first transition from `nominal` to `fair` occurs during measured run 3,
  Polish `shared_batched`. The complete first two measured runs remain nominal:
  legacy/sequential 28.1982, shared/sequential 24.1656, legacy/batched 28.8947,
  shared/batched 24.6607 sustained tokens/s. The matrix regression therefore
  already exists before that transition.

Pooling arbitrary nominal-only captures would unbalance the prompt mix because
legacy/batched has five such captures while the other configurations have four.
The comparison above instead uses complete runs 1–2. Snapshots cannot establish
constant clocks between observations. This diagnostic also does not perform a
fresh ordinary-target run; its historical matching IDs are supporting evidence,
not a replacement for `mtp-bench --compare`.

## Remaining cost to reach 32

Legacy/batched spends 30.3454 seconds on the GPU out of 35.6149 seconds of
decode (85.20%). CPU encoding is 0.2207 seconds (0.62%), and draft work is
4.0301 seconds (11.32%). GPU time overlaps CPU completion-wait time; they must
not be added.

The two-prompt weighted rate needs 10.85% less total decode time to reach 32,
or 12.57% less target execution time if acceptance and other costs stay fixed.
The Polish prompt is harder: its weighted rate is 25.3273, and it needs
**24.17% less target execution time** under the same assumptions. Even removing
all draft work would yield only 28.43 for that prompt. Removing every cost except
target execution would yield only 29.36. Further target GPU work is necessary.

The code median exceeds 32, but the final `fair` run is 31.5275. Neither one
prompt's median nor the two-prompt aggregate establishes the mixed-use goal.

The next bounded experiment varies the number of output rows computed by each
SIMD group in the original matrix schedule, while retaining its per-output
FP32 arithmetic order. Fewer rows reduce live accumulators and packed weights
but require more SIMD groups and repeated input loads; more rows make the
opposite tradeoff. Previously rejected row-2 tests used the shared-input
schedule; they did not test this original schedule. New variants remain outside
production until target-machine measurements justify using them.

The standalone `block_legacy_tune` example compares row counts 1/2/4/8 and
threadgroup sizes 32/64 against the frozen original R4/T64 implementation.
It uses B3 BF16 and the up, down and vocabulary projection dimensions. All
output bits must match the baseline, with additional independent FP64 checks
before and after timing. Samples alternate AB/BA and exclude warmups.

```sh
git pull --ff-only
cargo build --release --locked --bin qwen-metal --example block_legacy_tune
./target/release/examples/block_legacy_tune \
  --sweep --iterations 32 > legacy-tiles.json
```

This does not load the trained target or MTP checkpoint. It records the source
fingerprints, raw GPU samples and power/thermal snapshots. Repeated synthetic
weights can benefit from cache, so the result cannot establish model tokens/s
or automatically select a production kernel. A candidate must subsequently be
measured in the full model on the M5.

## Local probe evidence

The [raw M1 sweep](benchmarks/m1-v0.6.3-legacy-tile-sweep.json) contains 24 paired
comparisons, each with 32 GPU samples per implementation and 10 excluded
warmups per implementation. Conditions are nominal throughout. R8/T64 results:

| B3 BF16 matrix | Original R4/T64 GPU time | Candidate R8/T64 GPU time | Ratio |
| --- | ---: | ---: | ---: |
| Up, 17408 × 5120 | 1.9932 ms | 1.3506 ms | 1.4758× |
| Down, 5120 × 17408 | 2.2657 ms | 1.4429 ms | 1.5702× |
| Vocabulary, 248320 × 5120 | 30.4740 ms | 21.9387 ms | 1.3891× |

R8/T32 is similar (1.4693× / 1.5803× / 1.3698×). R1/R2 are slower. The
R4/T64 control ratios are 1.012× / 1.039× / 1.001×, substantially smaller than
the R8 gains. These are isolated repeated-weight results on M1, not M5 model
throughput. R8 is **not integrated into or selected by the engine** in 0.6.3.

The release self-test passes 256 cases: all eight row/thread configurations,
B1–B4, F32/BF16 metadata and widths 512/1536/5120/17408. Every small-fixture
row is checked against FP64; large timed matrices check 32 distributed rows
per RHS. All output bits match the frozen baseline across all self-tests and
all 24 timed pairs. Buffer guards, input immutability, row divisibility,
32-lane SIMD and finite GPU timestamps are checked.

Exact embedded Metal source SHA-256:
`07f09997455129075b990a21adac1cd91ffbdd453ab8f3d00faf2e61cf08dd56`.
Harness source SHA-256:
`98c53ceecbe7a374ce82358a8db89a6ea194641ffe3ce91ec547acdd2f18d5b0`.

The final 0.6.3 all-target suite passes **120 unique tests**, with zero failures
and zero ignored tests when real Metal is enabled. This includes the CLI check
that removes explicit block-mode environment settings and confirms the new
legacy/batched default, matching ordinary-target greedy output and correct
decode timing counts. Explicit sequential and shared choices remain covered.
The 256-case standalone probe self-test is additional to that suite.
The release build and eight-capture tiny-model CLI comparison also pass with
block-mode overrides removed: the selected default is legacy/batched and
greedy IDs agree with ordinary target generation. Formatting and diff checks
pass. The existing `block` 0.1.6 dependency future-compatibility warning remains.
