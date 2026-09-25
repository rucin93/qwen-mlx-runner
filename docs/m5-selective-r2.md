# M5 row-tile results and selective R2 model comparison

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

## Evidence and limits

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
though 14 of 16 pairs are faster. Down is faster in all 32 pairs. A full-model
test is necessary to evaluate sustained behavior and outliers.

If these median differences applied unchanged to 64 layers, two gate/up and
one down projection per layer would save about **4.61 ms per width-3 target
round**. Paired mean differences instead imply 4.16 ms. This estimate assumes
the same memory/cache behavior and does not include width-1/2 tails. It is
roughly 6% of the previous target GPU time, and **does not establish 32 tok/s**
or close the weaker Polish prompt's approximately 24% target-time gap by itself.

## Opt-in integration in 0.6.4

`QWEN_METAL_BLOCK_MATMUL=mlp-r2` selects the measured two-row schedule only when
all of these conditions hold:

- Target block matrix multiplication uses aligned affine Q4/group64 weights.
- Metadata is stored in exact BF16 form.
- Block width is exactly three.
- Matrix shape is exactly 17408 × 5120 or 5120 × 17408.

Other dimensions, the vocabulary head, other formats and shorter/taller blocks
retain the earlier kernels. Single-token target decoding and MTP adapter
execution are unchanged. The normal default remains **legacy + batched
DeltaNet** until full-model M5 evidence supports another choice.

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
