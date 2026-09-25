# Synthetic Qwen3.8-27B decode benchmark

`Engine::synthetic_qwen27b(context, history)` constructs the full 64-layer text
decode graph using the [published Qwen3.8-27B text dimensions](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).
Its 48 DeltaNet layers and 16 full-attention layers execute the same forward
path as a loaded checkpoint, including embedding, MLP, output head, recurrent
state, and attention over the requested history length. It needs no checkpoint
file or network access.

**This is synthetic reused-weight zero-history performance, not real-model
token performance or output quality.** Each projection role has one deterministic
Q4 group-64 weight matrix reused by every matching layer. The real model has
distinct layer weights, so this working set is much smaller and may benefit
from GPU caching. The `history` slots in every attention cache contain zeros;
the constructor advances the position without performing prompt prefill.
Only the measured decode steps write real KV entries. Output values are finite
test data, not meaningful token predictions.

The constructor keeps the model's untied embedding and output head as separate
synthetic buffers. At context 2,048, unique weights occupy about 1.764 GiB,
DeltaNet state about 0.146 GiB, and attention KV caches 0.250 GiB. KV memory
grows by 0.125 GiB per additional 1,024 context slots. The constructor checks
Metal's recommended working set before allocating and validates that history is
smaller than context. `QWEN_METAL_REFERENCE=1` selects the existing reference matvec kernels
for same-binary comparisons.
