// Original parallel RMS normalization, with FP32 inputs and arithmetic.
// ABI: x,w,y at buffers 0..2; uint parameters at 15: length, epsilon bits.
// HOST PRECONDITIONS: exactly one 256-thread group, 32-thread SIMD groups,
// nonzero length, valid epsilon, and disjoint output/input buffers.
// Buffers start at aligned Metal-buffer offsets; no padding is read or written.
// Eight SIMD groups reduce their partial sums through threadgroup memory.
// The reduction order differs from the single-SIMD reference implementation.
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_parallel(device const float *x [[buffer(0)]],
                              device const float *w [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant uint *p [[buffer(15)]],
                              uint tid [[thread_position_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partials[8];
    threadgroup float inverse_rms;

    const uint length = p[0];
    const uint vectors = length / 4;
    device const float4 *x4 = reinterpret_cast<device const float4 *>(x);
    float4 squares = 0.0f;
    for (uint i = tid; i < vectors; i += 256) {
        const float4 values = x4[i];
        squares += values * values;
    }
    float sum = (squares.x + squares.y) + (squares.z + squares.w);
    // At most three scalar elements remain. Assign them to the first threads.
    const uint tail = vectors * 4 + tid;
    if (tid < length % 4) sum += x[tail] * x[tail];

    const float simd_total = simd_sum(sum);
    if (lane == 0) partials[tid / 32] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread participates in its SIMD reduction and both barriers.
    // Only SIMD group zero receives the eight initialized partial sums.
    const float total = simd_sum(tid < 8 ? partials[tid] : 0.0f);
    if (tid == 0) {
        inverse_rms = rsqrt(total / float(length) + as_type<float>(p[1]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inverse = inverse_rms;
    device const float4 *w4 = reinterpret_cast<device const float4 *>(w);
    device float4 *y4 = reinterpret_cast<device float4 *>(y);
    for (uint i = tid; i < vectors; i += 256) {
        y4[i] = (x4[i] * inverse) * w4[i];
    }
    if (tid < length % 4) y[tail] = (x[tail] * inverse) * w[tail];
}
