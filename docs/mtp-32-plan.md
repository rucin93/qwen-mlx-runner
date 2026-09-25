# Native MTP path toward 32 tokens/s

The requested target is approximately 32 generated tokens/s for one mixed-use
chat on M5 Pro / 48 GB, on external power, using the existing Qwen3.8-27B Q4
checkpoint and this project's own Rust + Metal runtime. The initial ordinary-
target baseline was 16.1308 tokens/s; the mixed-use 32 tokens/s goal remains unmet.

The user's [audited 0.6.4 full-model comparison](m5-selective-r2.md) measures
**28.6868 sustained tokens/s for selective R2 versus 27.8079 for legacy MTP
(+3.1608%)** over five prompts. All 15 measured R2/legacy pairs improve, with
matching ordinary-target IDs, text and finish reasons. Six complete nominal
triples retain a +2.8926% gain; the full run transitions from nominal to fair.
Only code reaches a 32 tokens/s median. Explanation and planning remain near
26 tokens/s; analysis and rewrite are near 29.

Version 0.6.5 selects the measured R2 route by default only for the exact Metal
device name `Apple M5 Pro`, with an explicit `QWEN_METAL_BLOCK_MATMUL=legacy`
override and legacy defaults elsewhere. Its eligible B3 BF16 Q4/group64 MLP
shapes, vocabulary-head fallback and batched DeltaNet are unchanged. The earlier
[shared-matrix regression](m5-kernel-comparison.md) and M5's rejection of the
M1-winning R8 tile are reasons to keep this choice device-specific.

Target GPU work still occupies **84.65%** of R2 decode. At unchanged acceptance
and other costs, reaching 32 requires approximately **22% less target GPU time
for explanation/planning** and 10% less for analysis/rewrite. CPU encoding and
commit account for about 0.67% of decode and cannot close that gap alone. The
next experiment must target GPU cost and establish a full-model benefit; the
current R2 result does not establish 32 tokens/s.

## Evidence and approach

The measured operation inventory implies 14,412,349,440 bytes of Q4 weights and
BF16 scale/bias metadata per single-token matrix pass. Apple's M5 Pro specification
lists 307 GB/s memory bandwidth. A streaming bandwidth-only limit of about
21.3 tokens/s motivates sharing each weight read across several verification
positions instead of expecting another 2x from single-vector kernel tuning.

The existing mlx-community target has no MTP tensors. The separate native
`mlx-community/Qwen3.8-27B-MTP-4bit` adapter at revision
`b643c01b6d3b094e325edb6ebd832e16c486c575` contains eight affine Q4/group64
matrices plus seven already-sanitized BF16 norm vectors, about 239 MB. It shares
the target embedding/head and has its own one-layer attention cache. The default
block size of three means one known target token plus two draft proposals.

## Implementation boundaries

1. **GPU batch and slice ABI** — `gpu.rs`, `matmul_block.metal`, numerical tests.
   Preserve the zero-offset path; add checked offsets and actual B=1..4 weight
   reuse inside matrix kernels. Cover dense F16 and affine Q4/Q8 with F32/BF16
   metadata. Include a bounded GPU copy operation for state snapshots.
2. **Target block transaction** — `engine/block.rs` and narrow Engine additions.
   Traverse layers before tokens; batch each linear map while retaining causal
   token order in DeltaNet/convolution and attention. Keep GPU snapshots of
   recurrent/conv state after every prefix, including zero. Commit any prefix
   without replaying the target. Mask rejected KV suffixes through position.
   Return logits and target hidden states after final RMS for every position.
3. **Speculative acceptance** — pure CPU `speculative.rs`.
   Preserve greedy target choices. For stochastic sampling, use the normalized
   target/draft distributions with acceptance min(1,p/q) and rejection sampling
   from normalized positive p-q, including the configured top-k/top-p semantics.
   Report accepted drafts and consumed inputs separately from emitted tokens.
4. **Native MTP** — dedicated loader and `engine/mtp.rs`.
   Validate dimensions/conventions; separately normalize embedding and target
   post-RMS hidden, concatenate in that order, project and run one full-attention
   block plus final norm/head. Keep draft cache separate. Prefill shifted token/
   hidden pairs, rewind rejected entries and seed the next cycle from verified
   hidden states. No external inference runtime or lossy weight conversion step.
5. **Integration and evidence** — optional CLI/server path plus a mixed-prompt
   benchmark. Preserve normal generation as a fallback/reference. Handle EOS,
   output limits, cancellation and state reset at accepted-prefix boundaries.
   Benchmark actual verified output tokens divided by total decode wall time,
   including draft, target, rollback and acceptance work. Record acceptance rates
   and compare greedy traces with ordinary target generation.

## Verification

Independent scalar/FP64 fixtures cover matrix batches and MTP outputs. Complete
hybrid model tests compare block and sequential logits from reset and nonempty
history, including every rollback prefix and subsequent continuation. Pure
acceptance tests cover rejection at each position and stochastic residual laws.
Run existing GPU and HTTP tests before publishing. Microbenchmarks are labeled
as primitives, never presented as achieved chat throughput. Final 32 tokens/s
acceptance requires measurements on the user's M5 using representative mixed
prompts; speculative acceptance is workload dependent.

## Primary sources

- [Apple M5 Pro specifications](https://www.apple.com/macbook-pro/specs/)
- [Native Q4 MTP adapter](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit/tree/b643c01b6d3b094e325edb6ebd832e16c486c575)
- [MTP architecture/cache reference](https://github.com/Blaizzy/mlx-vlm/blob/ad4a3ccd6483aa2db4d84b70108aca103c6a9b15/mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py)
- [Speculative decoding sampling law](https://arxiv.org/abs/2211.17192)

These sources document formats and algorithms. Implementation remains original
Rust/Metal code; their inference runtimes are not linked or invoked.

## Next architectural experiment

The measured selective R2 gain leaves a substantial gap on the weaker prompts.
A subsequent bounded experiment should evaluate
Metal 4 TensorOps within our own shaders. Apple recommends this API for custom
matrix workloads targeting M5's GPU Neural Accelerators, while noting that
skinny decode matrices can remain bandwidth limited.
[Apple M5 GPU guidance](https://developer.apple.com/videos/play/tech-talks/111432/).

Apple also documents custom quantization and cooperative tensor inputs.
[Metal tensor operations](https://developer.apple.com/videos/play/wwdc2026/330/).
This offers a route for keeping the existing packed checkpoint and custom
dequantization inside the Rust/Metal engine. Version 0.6.5 adds an isolated
[FP32 TensorOps probe](tensorops-probe.md), outside the inference runtime.
FP32 operand support, successful compilation on the target OS/SDK, numerical
error and actual B3 speed must be evaluated separately. A matrix primitive may
change reduction order even with FP32 inputs; it cannot inherit the scalar
kernel's bitwise-equality claim. No FP16 conversion or weight requantization
should be treated as an invisible optimization. Retain the measured scalar
fallback and validate real greedy traces before choosing any new path.
