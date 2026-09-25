// Original FP32 attention-value reduction. Same buffers and parameters as
// attn_values: scores[H,L], cache[L,KVH,D], gate[H,D], output[H,D].
// Each 256-thread group computes one head and 32 adjacent output channels.
// Eight SIMD groups divide the history, retaining coalesced cache loads.
// Host dispatches exactly H*ceil(D/32) groups, with disjoint output buffers.
#include <metal_stdlib>
using namespace metal;

kernel void attn_values_parallel(device const float *scores [[buffer(0)]],
                                 device const float *cache [[buffer(1)]],
                                 device const float *gate [[buffer(2)]],
                                 device float *output [[buffer(3)]],
                                 constant uint *p [[buffer(15)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float partials[256];
    const uint heads = p[0], kv_heads = p[1], dim = p[2], length = p[3];
    const uint tiles = dim / 32 + uint(dim % 32 != 0);
    const uint head = group / tiles;
    const uint channel = (group % tiles) * 32 + tid % 32;
    const uint partition = tid / 32;
    const uint kv_head = head / (heads / kv_heads);
    float sum = 0.0f;
    if (channel < dim) {
        for (uint t = partition; t < length; t += 8) {
            sum += scores[head * length + t]
                * cache[(t * kv_heads + kv_head) * dim + channel];
        }
    }
    // Tail channels also initialize their slot and participate in the barrier.
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (partition == 0 && channel < dim) {
        const uint lane = tid % 32;
        float total = 0.0f;
        for (uint part = 0; part < 8; ++part) total += partials[part * 32 + lane];
        const uint index = head * dim + channel;
        output[index] = total / (1.0f + exp(-gate[index]));
    }
}
