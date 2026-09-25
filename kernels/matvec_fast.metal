// Original affine-quantized GEMV, specialized to eliminate per-weight division.
// ABI matches matvec_affine: w,s,b,x,y at buffers 0..4; uint params at 15:
// rows,cols,bits,group. Caller must select the matching bits/group specialization.
// Preconditions: cols is a multiple of GROUP; SIMD width is 32.
// Main kernels: four rows/SIMD, grid ceil(rows/4)*32, threadgroup 128 threads.
// _r1 kernels: one row/SIMD, grid rows*32, threadgroup 128 threads.
// Padding the grid to whole threadgroups is safe. No shared-memory allocation.
#include <metal_stdlib>
using namespace metal;

// Each lane owns one complete packed word. Its input values are reused for
// four independent rows. Adjacent lanes load adjacent packed words, and each
// packed word and affine parameter pair is loaded only once per row/iteration.
// Affine evaluation is factored per word: s*sum(q*x) + b*sum(x).
// This changes FP32 summation order relative to elementwise dequantization.
template <uint BITS, uint GROUP, uint ROWS>
inline void affine_gemv_packed(device const uint *w,
                              device const float *scales,
                              device const float *bias,
                              device const float *x,
                              device float *y,
                              constant uint *p,
                              uint gid, ushort lane) {
    constexpr uint PACK = 32 / BITS;
    constexpr uint WORDS_PER_GROUP = GROUP / PACK;
    const uint first_row = (gid / 32) * ROWS;
    if (first_row >= p[0]) return; // SIMD-uniform, including padded dispatch.
    const uint words_per_row = p[1] / PACK;
    const uint groups_per_row = p[1] / GROUP;
    float totals[ROWS];
    #pragma unroll
    for (uint r = 0; r < ROWS; ++r) totals[r] = 0.0f;

    for (uint word_index = lane; word_index < words_per_row; word_index += 32) {
        const uint col = word_index * PACK;
        const uint group_index = word_index / WORDS_PER_GROUP;
        const float4 xa = *reinterpret_cast<device const float4 *>(x + col);
        float4 xb = 0.0f;
        if (BITS == 4) xb = *reinterpret_cast<device const float4 *>(x + col + 4);
        const float xsum = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
        #pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            const uint row = first_row + r;
            if (row < p[0]) {
                const uint packed = w[row * words_per_row + word_index];
                const uint gi = row * groups_per_row + group_index;
                float qdot;
                if (BITS == 4) {
                    const float4 qa = float4(packed & 15u, (packed >> 4) & 15u,
                                             (packed >> 8) & 15u, (packed >> 12) & 15u);
                    const float4 qb = float4((packed >> 16) & 15u, (packed >> 20) & 15u,
                                             (packed >> 24) & 15u, packed >> 28);
                    qdot = dot(qa, xa) + dot(qb, xb);
                } else {
                    const float4 qa = float4(packed & 255u, (packed >> 8) & 255u,
                                             (packed >> 16) & 255u, packed >> 24);
                    qdot = dot(qa, xa);
                }
                totals[r] += fma(scales[gi], qdot, bias[gi] * xsum);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < ROWS; ++r) {
        const float total = simd_sum(totals[r]);
        if (lane == 0 && first_row + r < p[0]) y[first_row + r] = total;
    }
}

#define AFFINE_GEMV_ENTRY(NAME, BITS, GROUP, ROWS) \
kernel void NAME(device const uint *w [[buffer(0)]], \
                 device const float *scales [[buffer(1)]], \
                 device const float *bias [[buffer(2)]], \
                 device const float *x [[buffer(3)]], \
                 device float *y [[buffer(4)]], \
                 constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    affine_gemv_packed<BITS, GROUP, ROWS>(w, scales, bias, x, y, p, gid, lane); \
}

AFFINE_GEMV_ENTRY(matvec_q4_g32, 4, 32, 4)
AFFINE_GEMV_ENTRY(matvec_q4_g64, 4, 64, 4)
AFFINE_GEMV_ENTRY(matvec_q4_g128, 4, 128, 4)
AFFINE_GEMV_ENTRY(matvec_q8_g32, 8, 32, 4)
AFFINE_GEMV_ENTRY(matvec_q8_g64, 8, 64, 4)
AFFINE_GEMV_ENTRY(matvec_q8_g128, 8, 128, 4)
AFFINE_GEMV_ENTRY(matvec_q4_g32_r1, 4, 32, 1)
AFFINE_GEMV_ENTRY(matvec_q4_g64_r1, 4, 64, 1)
AFFINE_GEMV_ENTRY(matvec_q4_g128_r1, 4, 128, 1)
AFFINE_GEMV_ENTRY(matvec_q8_g32_r1, 8, 32, 1)
AFFINE_GEMV_ENTRY(matvec_q8_g64_r1, 8, 64, 1)
AFFINE_GEMV_ENTRY(matvec_q8_g128_r1, 8, 128, 1)

#undef AFFINE_GEMV_ENTRY
