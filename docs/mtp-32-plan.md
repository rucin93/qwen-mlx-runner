# Native MTP path toward 32 tokens/s

The requested target is approximately 32 generated tokens/s for one mixed-use
chat on M5 Pro / 48 GB, on external power, using the existing Qwen3.8-27B Q4
checkpoint and this project's own Rust + Metal runtime. The observed baseline
remains 16.1308 tokens/s; this document does not claim the new target is achieved.

Follow-up: the user's 0.6.0 block-3 report now establishes 26.7659 sustained
tokens/s versus 16.6282 for its paired sequential baseline, with matching greedy
traces. [The audited budget](m5-mtp-v0.6.md) identifies target execution as 86.78%
of decode time. The next bounded changes are shared Q4 unpacking for B2/B3 and
register-resident multi-step DeltaNet with only necessary rollback snapshots.
The later block-4 report has Low Power Mode enabled and cannot be compared
directly to the block-3 result with it disabled.

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
