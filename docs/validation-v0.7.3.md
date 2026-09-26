# v0.7.3 validation and remaining target checks

Date: 2026-09-26. Local host: Apple M1, 16 GB unified memory.
This report covers the OpenCode latency diagnostics and MTP prompt-prefill
changes in 0.7.3, including the reasoning integration introduced in 0.7.2.

## Completed local validation

The full suite ran with actual Metal GPU access and temporary loopback HTTP
listeners:

```sh
OPENCODE_SDK_DIR="$PWD/results/opencode-sdk" \
  cargo test --offline --locked --all-targets -- \
  --include-ignored --test-threads=1
cargo fmt --all -- --check
git diff --check
```

**190 distinct tests passed, zero failed and zero were ignored.** This is 137
library tests, 12 binary tests and 41 integration tests. One reference-path
integration test re-enters itself in a child process; its successful child run
is not counted as an additional distinct test. Example targets also compiled.
The complete local log is `results/openai-schema/v0.7.3-full-tests.log`;
`results/` is intentionally excluded from Git.

`cargo build --release --offline --locked -j 2` also passed. The resulting
executable reports `qwen-metal 0.7.3`. A release smoke test generated the same
eight-token text with ordinary generation and B3 MTP on `tiny-q4` / `tiny-mtp`,
with `finish_reason="length"` in both runs. Logs are retained locally as
`results/openai-schema/v0.7.3-release-build.log` and
`results/openai-schema/v0.7.3-release-smoke.json`. Cargo reports the existing
future-compatibility warning from the transitive `block` 0.1.6 dependency;
it did not prevent compilation or testing.

Coverage includes:

- Target prefill on Q4 and BF16 fixtures, all block widths 1–4 and partial tails:
  bit-identical hidden states, final prompt logits and subsequent generation
  logits compared with the previous verification-block path.
- A dispatch audit showing that nonfinal target prompt blocks omit exactly the
  vocabulary projection while preserving the remaining dispatches.
- Draft prefill: bit-identical K/V caches and future hidden states/logits,
  including truncation, replacement of suffixes, invalid inputs, capacity and
  reset. The query, attention output and MLP are absent from cache-only prefill.
- Cancellation during partial prefill and identical output on the next request,
  for every supported MTP block width.
- Existing independent scalar/FP64 matrix and hybrid-model fixtures, block
  rollback, sampling, HTTP context budgets and tool-call behavior.
- Separate HTTP reasoning/content/tool readiness fields in streaming and
  non-streaming responses, preserving existing prefill/decode timings.
- The actual compatible SDK and installed OpenCode CLI against isolated HTTP
  fixtures. OpenCode executes a read of a temporary README, preserves reasoning
  in tool-result history and completes exactly two main requests with the
  auxiliary title agent disabled.

An independent read-only review of the complete branch against 0.7.1 found no
actionable correctness findings. The review included both the 0.7.2 reasoning
configuration/contracts and the 0.7.3 production changes.

The installed OpenCode 1.15.12 still exhibits the documented CLI event-drain
limitation: it can exit before printing final NDJSON reasoning/text events,
although the complete result is persisted in its session. The test separately
checks the live prefix and exported session. This does not establish a complete
live CLI tail or visual TUI behavior. See the [compatibility contract](openai-compatibility.md).

## What remains unverified

No full Qwen3.8-27B checkpoint or MTP adapter is available on this local host.
The local fixtures are small and untrained. These tests do not establish:

- Independent trained-model logits, answer quality or coding-agent reliability.
- A v0.7.3 M5 Pro prefill/first-answer latency improvement.
- Achievement of the mixed-use 32 sustained tokens/s target.

The latest full-model throughput evidence remains the user-supplied 0.6.4 M5
comparison: **28.6868 sustained tokens/s with selective R2**, versus 27.8079
with legacy MTP. Only the code prompt exceeds 32 tokens/s. The new prefill
shortcuts reduce work before decoding; they are not evidence of a higher
sustained decode rate. [Audited M5 evidence](m5-selective-r2.md).

## Next run on the M5 Pro

Use the target M5 Pro / 48 GB, external power, Low Power Mode off, and no
concurrent inference. Keep both existing checkpoint directories unchanged.
Build 0.7.3 on that Mac and record the commit alongside the result:

```sh
cargo build --release --locked
mkdir -p results/v0.7.3
git rev-parse HEAD > results/v0.7.3/commit.txt
./target/release/qwen-metal --version > results/v0.7.3/version.txt

QWEN_METAL_REFERENCE=0 QWEN_METAL_GEMV=aligned \
QWEN_METAL_NORM=parallel QWEN_METAL_METADATA=bf16 \
QWEN_METAL_ATTN_VALUES=serial \
  ./target/release/qwen-metal mtp-bench \
  --model models/Qwen3.8-27B-4bit \
  --mtp models/Qwen3.8-27B-MTP-4bit \
  --context 8192 --max-tokens 128 --runs 3 --block-size 3 \
  --temperature 0 --compare-mlp-r2 \
  > results/v0.7.3/mtp-mixed-greedy.json
```

Run this only after confirming that the executable reports `qwen-metal 0.7.3`.
The existing harness loads the target and adapter once, rotates ordinary
generation / legacy MTP / selective R2, excludes warmups and records exact
output IDs, finish reasons, phase timings and machine-condition snapshots.
Its 32-token/s gate requires all five prompt medians to pass and both MTP
variants to match ordinary generation. See the [full acceptance criteria](m5-selective-r2.md#one-load-full-model-comparison).

For actual OpenCode latency, use `QWEN_METAL_LOG_REQUESTS=1` on the existing
server launch, retain the same prompt/context/reasoning effort, and inspect
`queue_seconds`, `prefill_seconds`, `first_reasoning_seconds` and
`first_content_seconds`. Preserve the resulting JSON timing records for
comparison; they contain no prompt or generated text. The benchmark above
does not reproduce OpenCode's system instructions, tools and history.

Independent full-model validation remains a separate check: compare the
unchanged checkpoint and tokenization with a trusted external reference, then
evaluate complete answers and real tool arguments. Agreement between two paths
in this engine is useful regression evidence but is not that independent
reference.
