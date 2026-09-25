# Rust/Metal implementation plan

User requested an independent engine, explicitly replacing the earlier backend
recommendation. Work in this empty repository; no existing changes to isolate.

- [x] Validate and load local dense hybrid checkpoints, with format tests.
- [x] Implement custom packed matvec, normalization, RoPE, conv, DeltaNet,
      attention and feed-forward Metal kernels, with numerical tests.
- [x] Wire an autoregressive forward pass with reusable GPU buffers and a sampler.
- [x] Implement text template/tokenization and streaming localhost chat server.
- [x] Exercise a complete tiny hybrid checkpoint through forward and chat;
      independently test the HTTP route contract and cancellation.
- [x] Add kernel and model benchmark commands and document exact limits.
- [x] Run formatting, compilation, tests and independent code review; fix findings.

## Remaining target validation and optimization

- [ ] Validate real Qwen3.8-27B checkpoint logits and chat outputs against a
      trusted reference on hardware with sufficient memory.
- [ ] Run fixed-workload benchmarks on the target M5 Pro / 48 GB.
- [ ] Profile and optimize the measured bottleneck. Current possible targets:
      batched prefill, packed matvec input reuse, fewer dispatches, mixed-precision
      cache, and a separately verified MTP implementation.

The completed initial implementation is not evidence that these remaining
performance and full-checkpoint requirements have been achieved.

The architecture document defines cross-component interfaces. Separate workers
own loader, kernels and serving; the primary owns engine, CLI and integration.
