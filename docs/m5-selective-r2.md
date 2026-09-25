# M5 selective R2: audited full-model result

The user-supplied 0.6.4 report, audited on **2026-09-25**, establishes a modest
full-model gain: **28.6868 sustained tokens/s for selective MLP R2 versus
27.8079 for legacy MTP (+3.1608%)**. All 15 measured prompt/run pairs improve.
This supports the 0.6.5 default policy for the exact Metal device name
`Apple M5 Pro`. **The five-prompt 32 tokens/s goal remains unmet.**

Raw evidence: [M5 Pro 0.6.4 three-variant comparison](benchmarks/m5-pro-v0.6.4-mlp-r2-comparison-user.json),
SHA-256 `4c5ea01347766c85f61c87e76b63cd335dc0cb4e4da95a972eab1f3c7214d1bf`.
It records one loaded target/adapter, block size 3, 192 eligible target matrices,
five prompts, one excluded warmup and three measured runs per prompt/variant.

## Full-model correctness and speed

The audit recomputed counts and rates from all 60 captures: 15 warmups and 45
measured captures. Every capture contains exactly **128 verified token IDs and
127 sustained decode tokens**, with `finish_reason="length"`. Both MTP variants
match ordinary target IDs, text and finish reason in all 20 prompt/run groups,
and match each other. The output traces and acceptance/round-width accounting
also match the [earlier 0.6.0 report](benchmarks/m5-pro-v0.6-mtp-b3-user.json).
These are exact 128-token prefix checks; they do not establish EOS behavior or
answer completeness.

| Measured aggregate | Ordinary target | Legacy MTP | Selective R2 MTP |
| --- | ---: | ---: | ---: |
| Sustained tokens/s, total sustained tokens / total decode time | 16.3033 | 27.8079 | **28.6868** |
| End-to-end tokens/s, including prefill | 11.6726 | 21.1372 | 21.8714 |
| Total decode seconds | 116.8472 | 68.5058 | 66.4068 |
| Target GPU seconds | 114.1476 | 58.3115 | 56.2147 |

R2's paired sustained-rate gains range from **2.3881% to 5.7676%**, with median
**2.9941%**. Target GPU time decreases in all 15 pairs and by **3.5957%** in
aggregate. The 2.0967 seconds saved on target GPU account for 99.89% of the
2.0990 seconds saved in decode. Both MTP variants have 1,125 accepted drafts,
1,551 proposals and 780 verification rounds; improved acceptance does not
explain the gain. R2 is 7.18% faster than historical 0.6.0 MTP, but that is a
cross-session/version comparison: only the current **+3.16%** isolates R2.

| Prompt | Legacy median sustained tokens/s | R2 median sustained tokens/s | R2 median decode time still to save for 32 tokens/s |
| --- | ---: | ---: | ---: |
| `pl-explanation` | 25.4435 | 26.1821 | 0.8819 s (18.18%) |
| `en-code` | 32.8591 | **33.9466** | Already meets the threshold |
| `pl-analysis` | 28.4177 | 29.2886 | 0.3674 s (8.47%) |
| `en-rewrite` | 28.3861 | 29.3602 | 0.3568 s (8.25%) |
| `pl-planning` | 25.4221 | 26.0292 | 0.9104 s (18.66%) |

At 127 sustained tokens, the 32 tokens/s budget is 3.96875 decode seconds per
run. Target GPU still occupies **84.65%** of R2 decode. Using mean phase costs
and holding acceptance and other work fixed, target GPU must shrink by about
**21.62% for explanation, 21.85% for planning, 10.10% for analysis and 9.79% for
rewrite**. Target CPU encoding and commit together occupy only 0.67% of decode.
Further target GPU work is the substantial remaining opportunity; this report
does not identify an individual kernel as the next bottleneck.

## Thermal and order controls

Each variant occupies each timing position once per prompt across the three
measured runs. Immediate predecessors are not balanced. All 90 measured power
snapshots have Low Power Mode off; these snapshots do not establish AC power
or equal GPU clocks.

The first thermal transition is `en-code`, run 2, legacy MTP (capture 35,
zero-based): nominal to fair. Later captures remain fair. Comparing only
**complete nominal triples** leaves all five run-1 prompts plus run-2
`pl-explanation`: six groups, with legacy at **27.4835** and R2 at **28.2785**
sustained tokens/s (**+2.8926%**, target GPU time down 3.3171%). Eight complete
fair triples retain **+3.3705%**, with target GPU time down 3.8399%. The remaining
triple crosses the thermal transition. Thus the R2 gain persists in both
matched subsets, although there are not three nominal runs per prompt.

Ordinary target `pl-planning` run 3 takes 9.4858 decode seconds, versus
7.6134/7.6850 in the earlier runs. It inflates the pooled R2/ordinary-target
speedup; the direct R2/legacy comparison avoids that outlier. Evidence remains
limited to this device, workload and short generation length.

## Earlier isolated row-tile evidence

The user's [complete M5 sweep](benchmarks/m5-pro-v0.6.3-legacy-tile-sweep-user.json)
rejects R8 and supports testing R2/T64 only on the two MLP matrix dimensions.
Its SHA-256 is `d316e60e2c412f4db6c9bf5a41e9cf8178b67fe932df73681ac10b5ab45c39d0`.
The source hashes match both the released harness and the earlier M1 sweep.
**These are isolated matrix measurements, not full-model throughput.**

| B3 BF16 matrix | Paired R4/T64 median | R2/T64 median | GPU ratio | Faster pairs |
| --- | ---: | ---: | ---: | ---: |
| Gate/up, 17408 × 5120 | 0.221417 ms | 0.205458 ms | **1.0777×** | 28/32 |
| Down, 5120 × 17408 | 0.267625 ms | 0.227479 ms | **1.1765×** | 32/32 |
| Vocabulary, 248320 × 5120 | 2.747333 ms | 2.758396 ms | 0.9960× | 8/32 |

R8 is slower in every one of its 192 paired samples. R8/T64 ratios are
0.8790× / 0.8398× / 0.9001× for these shapes. Its M1 improvement does not transfer
to M5, so R8 remains outside the runtime. R2 is also inappropriate for the
vocabulary matrix. T32 adds no consistent advantage over T64.

## Isolated sweep checks

All 24 shape/tile/thread configurations have zero differing output bits and
valid FP64 checks. All 1,536 recorded GPU durations are positive and finite;
reported medians and ratios reproduce from the raw arrays. All 48 power/thermal
snapshots have Low Power Mode off and nominal thermal state. The R4/T64 controls
are near unity: 1.0010× / 1.0003× / 1.0032×.

The initial gate/up baseline changes from 0.4087 to roughly 0.219 ms across the
sweep. This shows why comparisons must use each candidate's contemporaneous
control, not a baseline from another part of the run. R2/T64 gate/up improves
in both halves of its own sample set, by paired medians, but has two late spikes
around 0.373–0.379 ms. Its second-half mean saving is consequently negative even
though 14 of 16 pairs are faster. Down is faster in all 32 pairs. The full-model
comparison above measures the resulting runtime benefit.

Applying the isolated median differences to 64 layers previously suggested
4.61 ms saved per width-3 target round (4.16 ms from paired means). That estimate
assumed unchanged memory/cache behavior and omitted width-1/2 tails. The actual
full-model target GPU saving is **2.69 ms per verification round on average**;
the full-model measurement determines the remaining budget above.

## Device default in 0.6.5 and explicit overrides

Version 0.6.4 introduced the opt-in route. The 0.6.5 default policy selects it
only for the exact Metal device name **`Apple M5 Pro`**; other devices retain
legacy matrices. `QWEN_METAL_BLOCK_MATMUL=legacy` explicitly restores legacy,
and `QWEN_METAL_BLOCK_MATMUL=mlp-r2` explicitly selects R2 regardless of the
device default. The measured two-row schedule is still eligible only when all
of these conditions hold:

- Target block matrix multiplication uses aligned affine Q4/group64 weights.
- Metadata is stored in exact BF16 form.
- Block width is exactly three.
- Matrix shape is exactly 17408 × 5120 or 5120 × 17408.

Other dimensions, the vocabulary head, other formats and shorter/taller blocks
retain the earlier kernels. Single-token target decoding and MTP adapter
execution are unchanged. **Batched DeltaNet** remains the default on all devices.

The block configuration is now represented by the typed matrix modes `legacy`,
`shared`, and `mlp_r2` in JSON, alongside `batched_delta`. The environment spelling
of the new option is `mlp-r2`. Archived reports preserve their earlier boolean
schema unchanged.

## One-load full-model comparison

The new `--compare-mlp-r2` benchmark runs ordinary sequential target generation,
legacy/batched MTP, and selective-R2/batched MTP on the same loaded target and
adapter. Each prompt/mode receives an excluded warmup. Three measured runs
rotate the mode order so each mode occupies every timing position once per
prompt. It records exact output IDs, GPU/CPU timing and conditions, summarizes
the modes separately and checks both MTP variants against ordinary generation.

```sh
git pull --ff-only
cargo build --release --locked
QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 QWEN_METAL_ATTN_VALUES=serial \
./target/release/qwen-metal mtp-bench \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit \
  --context 8192 --max-tokens 128 --runs 3 --block-size 3 \
  --temperature 0 --compare-mlp-r2 > mlp-r2-model.json
```

Use external power with Low Power Mode off and no concurrent inference. This
command selects all configurations itself, so no block-matrix environment
override is needed. Do not add the mutually exclusive `--compare` or
`--compare-block-kernels` flags. The default workload contains all five mixed
prompts; `--prompts docs/kernel-ab-prompts.json` limits it to the two diagnostic
prompts if a narrower investigation is desired.

The report exposes `mlp_r2_eligible_target_matrices` based on the actual loaded
weights and `candidate_applicable`. A zero-eligible checkpoint still produces
timing and correctness data, but cannot confirm the R2 goal. Confirmation also
requires at least three measured runs, every candidate prompt median at least
32 sustained tokens/s, at least 64 sustained tokens per candidate run, complete
capture coverage and matching IDs/text/finish reasons against ordinary target
generation. It applies only to the recorded workload and conditions.

## Local validation

The final 0.6.4 all-target suite passes **127 unique tests**, zero failed or
ignored with real Metal enabled on M1. The production R2 route is tested on
both complete MLP shapes, including nonzero aligned buffer views and guards.
All **67,584 output values** match both the frozen `ef38979` R4 kernel and the
current Legacy implementation bit for bit; **192 independent FP64 row/RHS
samples** also pass. Dispatch-scope tests cover unsupported batches, formats,
shapes, alignments and the vocabulary-head fallback.

The three-mode tiny-model CLI test covers 24 captures, 16 ordinary-target
comparisons and eight R2/Legacy comparisons, checks separate summaries and
rotating order, and rejects goal confirmation for short/zero-eligible fixtures.
Those fixtures exercise the generation/report path; the full-size matrix tests
exercise actual R2 GPU dispatch. No full trained 27B target is run locally.
The release build and the same 24-capture public CLI smoke also pass: both
ordinary-target and candidate/Legacy comparisons agree, summaries recompute
from raw captures, and zero eligible matrices correctly yield
`candidate_applicable=false` and `goal_32_tps_confirmed=false`. Formatting and
diff checks pass. The existing `block` 0.1.6 dependency future-compatibility
warning remains.
