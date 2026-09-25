// Original affine GEMV with losslessly stored BF16 metadata, enabled by opt-in.
// Metadata ABI: scales/biases are ushort arrays containing the high 16 bits
// of their original FP32 values. Host compaction is legal only if EVERY
// original value has zero low 16 bits; retain the existing finite-value checks.
// Conversion reconstructs FP32 bits directly, with no half arithmetic or
// half-subnormal conversion. Arithmetic follows the library's FP32 math mode.
// Buffers 0..4: packed uint weights, ushort scales, ushort biases, float x,y.
// Parameters at buffer 15: rows,cols,bits,group; SIMD width must be 32.
// Packed variants: four rows/SIMD, grid ceil(rows/4)*32, 128-thread groups.
// The aligned Q4/g64 variant requires rows%4==0, cols%512==0, signed-int-safe
// dimensions/index products; grid (rows/4)*32 and 64-thread groups.
// Embedding uses buffers 0..3: weights,scales,biases,y; parameters are
// row,cols,bits,group; grid cols. All parameter/buffer validation is host-owned.
#include <metal_stdlib>
using namespace metal;

inline float affine_bf16_value(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

template <uint BITS, uint GROUP>
inline void affine_bf16_packed(device const uint *w,
                               device const ushort *scales,
                               device const ushort *bias,
                               device const float *x,
                               device float *y,
                               constant uint *p,
                               uint gid, ushort lane) {
    constexpr uint PACK = 32 / BITS;
    constexpr uint WORDS_PER_GROUP = GROUP / PACK;
    const uint first_row = (gid / 32) * 4;
    if (first_row >= p[0]) return;
    const uint words_per_row = p[1] / PACK;
    const uint groups_per_row = p[1] / GROUP;
    float totals[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint wi = lane; wi < words_per_row; wi += 32) {
        const uint col = wi * PACK;
        const uint group_index = wi / WORDS_PER_GROUP;
        const float4 xa = *reinterpret_cast<device const float4 *>(x + col);
        float4 xb = 0.0f;
        if (BITS == 4) xb = *reinterpret_cast<device const float4 *>(x + col + 4);
        const float xsum = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const uint row = first_row + r;
            if (row < p[0]) {
                const uint packed = w[row * words_per_row + wi];
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
                const float scale = affine_bf16_value(scales[gi]);
                const float offset = affine_bf16_value(bias[gi]);
                totals[r] += fma(scale, qdot, offset * xsum);
            }
        }
    }
    #pragma unroll
    for (uint r = 0; r < 4; ++r) {
        const float total = simd_sum(totals[r]);
        if (lane == 0 && first_row + r < p[0]) y[first_row + r] = total;
    }
}

#define AFFINE_BF16_PACKED_ENTRY(NAME, BITS, GROUP) \
kernel void NAME(device const uint *w [[buffer(0)]], \
                 device const ushort *scales [[buffer(1)]], \
                 device const ushort *bias [[buffer(2)]], \
                 device const float *x [[buffer(3)]], \
                 device float *y [[buffer(4)]], \
                 constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    affine_bf16_packed<BITS, GROUP>(w,scales,bias,x,y,p,gid,lane); \
}

AFFINE_BF16_PACKED_ENTRY(matvec_q4_g32_bf16, 4, 32)
AFFINE_BF16_PACKED_ENTRY(matvec_q4_g64_bf16, 4, 64)
AFFINE_BF16_PACKED_ENTRY(matvec_q4_g128_bf16, 4, 128)
AFFINE_BF16_PACKED_ENTRY(matvec_q8_g32_bf16, 8, 32)
AFFINE_BF16_PACKED_ENTRY(matvec_q8_g64_bf16, 8, 64)
AFFINE_BF16_PACKED_ENTRY(matvec_q8_g128_bf16, 8, 128)

#undef AFFINE_BF16_PACKED_ENTRY

inline float affine_bf16_u16_dot4(ushort q, float4 x) {
    return dot(float4(ushort4(q & ushort(0x000f), q & ushort(0x00f0),
                             q & ushort(0x0f00), q & ushort(0xf000))), x);
}

inline float affine_bf16_u16_dot16(ushort4 q, float4 xa, float4 xb,
                                  float4 xc, float4 xd) {
    return affine_bf16_u16_dot4(q.x, xa) + affine_bf16_u16_dot4(q.y, xb)
         + affine_bf16_u16_dot4(q.z, xc) + affine_bf16_u16_dot4(q.w, xd);
}

kernel void matvec_q4_g64_aligned_bf16(device const ushort *w [[buffer(0)]],
                                     device const ushort *scales [[buffer(1)]],
                                     device const ushort *bias [[buffer(2)]],
                                     device const float *x [[buffer(3)]],
                                     device float *y [[buffer(4)]],
                                     constant uint *p [[buffer(15)]],
                                     uint gid [[thread_position_in_grid]],
                                     ushort lane [[thread_index_in_simdgroup]]) {
    const int row = (int(gid) / 32) * 4;
    if (row >= int(p[0])) return;
    const int cols = int(p[1]);
    const int packed_stride = cols / 4;
    const int group_stride = cols / 64;
    const int tiles = cols / 512;
    const int weight_base = row * packed_stride;
    const int scale_base = row * group_stride;
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    for (int tile = 0; tile < tiles; ++tile) {
        const int col = tile * 512 + int(lane) * 16;
        const float4 a = *reinterpret_cast<device const float4 *>(x + col);
        const float4 b = *reinterpret_cast<device const float4 *>(x + col + 4);
        const float4 c = *reinterpret_cast<device const float4 *>(x + col + 8);
        const float4 d = *reinterpret_cast<device const float4 *>(x + col + 12);
        const float4 combined = a + b + c + d;
        const float xsum = combined.x + combined.y + combined.z + combined.w;
        const float4 factors = float4(1.0f, 0x1p-4f, 0x1p-8f, 0x1p-12f);
        const float4 xa = a * factors, xb = b * factors;
        const float4 xc = c * factors, xd = d * factors;
        const int wi = weight_base + col / 4;
        const int si = scale_base + col / 64;
        const ushort4 q0 = *reinterpret_cast<device const ushort4 *>(w + wi);
        sum0 += fma(affine_bf16_value(scales[si]),
                    affine_bf16_u16_dot16(q0, xa, xb, xc, xd),
                    affine_bf16_value(bias[si]) * xsum);
        const ushort4 q1 = *reinterpret_cast<device const ushort4 *>(w + wi + packed_stride);
        sum1 += fma(affine_bf16_value(scales[si + group_stride]),
                    affine_bf16_u16_dot16(q1, xa, xb, xc, xd),
                    affine_bf16_value(bias[si + group_stride]) * xsum);
        const ushort4 q2 = *reinterpret_cast<device const ushort4 *>(w + wi + 2 * packed_stride);
        sum2 += fma(affine_bf16_value(scales[si + 2 * group_stride]),
                    affine_bf16_u16_dot16(q2, xa, xb, xc, xd),
                    affine_bf16_value(bias[si + 2 * group_stride]) * xsum);
        const ushort4 q3 = *reinterpret_cast<device const ushort4 *>(w + wi + 3 * packed_stride);
        sum3 += fma(affine_bf16_value(scales[si + 3 * group_stride]),
                    affine_bf16_u16_dot16(q3, xa, xb, xc, xd),
                    affine_bf16_value(bias[si + 3 * group_stride]) * xsum);
    }
    sum0 = simd_sum(sum0); sum1 = simd_sum(sum1);
    sum2 = simd_sum(sum2); sum3 = simd_sum(sum3);
    if (lane == 0) {
        y[row] = sum0; y[row + 1] = sum1;
        y[row + 2] = sum2; y[row + 3] = sum3;
    }
}

kernel void embed_affine_bf16(device const uint *w [[buffer(0)]],
                              device const ushort *scales [[buffer(1)]],
                              device const ushort *bias [[buffer(2)]],
                              device float *y [[buffer(3)]],
                              constant uint *p [[buffer(15)]],
                              uint col [[thread_position_in_grid]]) {
    if (col >= p[1]) return;
    const uint pack = 32 / p[2];
    const uint word = w[p[0] * (p[1] / pack) + col / pack];
    const uint q = (word >> ((col % pack) * p[2])) & ((1u << p[2]) - 1u);
    const uint gi = p[0] * (p[1] / p[3]) + col / p[3];
    y[col] = float(q) * affine_bf16_value(scales[gi]) + affine_bf16_value(bias[gi]);
}
