// Original small-batch matrix multiplication for speculative verification.
// X and Y are token-major: X[b*cols+c], Y[b*rows+r]. Each SIMD group
// computes four rows for ALL B RHS. A packed weight and its metadata are loaded
// once and retained while multiplying each RHS; this is not a grid of GEMVs.
// B=1..4, Q4/Q8 groups32/64/128, FP32 or losslessly stored BF16 metadata.
// Generic paths support scalar-aligned buffer views, column and row tails.
// Host validates all index products and dispatches ceil(rows/4)*32 threads.
#include <metal_stdlib>
using namespace metal;

inline float block_metadata(float value) { return value; }
inline float block_metadata(ushort bits) { return as_type<float>(uint(bits) << 16); }

// Scalar construction deliberately permits X views aligned to 4 rather than
// 16 bytes. Apple buffers remain aligned; only the dynamic view can be offset.
inline float4 block_load4(device const float *x) {
    return float4(x[0], x[1], x[2], x[3]);
}

template<uint B, uint BITS, uint GROUP, typename M>
inline void block_affine(device const uint *w, device const M *scales,
                         device const M *bias, device const float *x,
                         device float *y, constant uint *p, uint gid, ushort lane) {
    constexpr uint PACK = 32 / BITS;
    constexpr uint WORDS_PER_GROUP = GROUP / PACK;
    const uint first_row = (gid / 32) * 4;
    if (first_row >= p[0]) return;
    const uint rows = p[0], cols = p[1];
    const uint words_per_row = cols / PACK, groups_per_row = cols / GROUP;
    float totals[B][4];
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) totals[b][r] = 0.0f;
    }
    for (uint wi = lane; wi < words_per_row; wi += 32) {
        const uint col = wi * PACK, group = wi / WORDS_PER_GROUP;
        uint packed[4];
        float scale[4], offset[4];
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            if (first_row + r < rows) {
                const uint gi = (first_row + r) * groups_per_row + group;
                packed[r] = w[(first_row + r) * words_per_row + wi];
                scale[r] = block_metadata(scales[gi]);
                offset[r] = block_metadata(bias[gi]);
            } else { packed[r] = 0; scale[r] = 0; offset[r] = 0; }
        }
        #pragma unroll
        for (uint b = 0; b < B; ++b) {
            const float4 xa = block_load4(x + b * cols + col);
            float4 xb = 0.0f;
            if (BITS == 4) xb = block_load4(x + b * cols + col + 4);
            const float xsum = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
            #pragma unroll
            for (uint r = 0; r < 4; ++r) {
                const uint q = packed[r];
                float qdot;
                if (BITS == 4) {
                    const float4 qa = float4(q & 15u, (q >> 4) & 15u, (q >> 8) & 15u, (q >> 12) & 15u);
                    const float4 qb = float4((q >> 16) & 15u, (q >> 20) & 15u, (q >> 24) & 15u, q >> 28);
                    qdot = dot(qa, xa) + dot(qb, xb);
                } else {
                    const float4 qa = float4(q & 255u, (q >> 8) & 255u, (q >> 16) & 255u, q >> 24);
                    qdot = dot(qa, xa);
                }
                totals[b][r] += fma(scale[r], qdot, offset[r] * xsum);
            }
        }
    }
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const float total = simd_sum(totals[b][r]);
            if (lane == 0 && first_row + r < rows) y[b * rows + first_row + r] = total;
        }
    }
}

// Aligned Q4/g64 fast path follows the existing single-token accumulation
// order exactly: 16 columns per lane, 512-column tiles, four rows per SIMD.
// It holds each tile's four packed words and metadata across every RHS while
// keeping only one RHS's 16 float inputs live. Host guarantees X % 16 = 0,
// W % 8 = 0, rows % 4 = 0, cols % 512 = 0 and signed-int-safe products.
inline float block_u16_dot4(ushort q, float4 x) {
    return dot(float4(ushort4(q & ushort(0x000f), q & ushort(0x00f0),
                             q & ushort(0x0f00), q & ushort(0xf000))), x);
}
inline float block_u16_dot16(ushort4 q, float4 a, float4 b, float4 c, float4 d) {
    return block_u16_dot4(q.x,a) + block_u16_dot4(q.y,b)
         + block_u16_dot4(q.z,c) + block_u16_dot4(q.w,d);
}
template<uint B, typename M>
inline void block_aligned_q4(device const ushort *w, device const M *scales,
                             device const M *bias, device const float *x,
                             device float *y, constant uint *p, uint gid, ushort lane) {
    const int row = (int(gid) / 32) * 4;
    if (row >= int(p[0])) return;
    const int rows = int(p[0]), cols = int(p[1]);
    const int packed_stride = cols / 4, group_stride = cols / 64;
    const int tiles = cols / 512;
    float totals[B][4];
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) totals[b][r] = 0.0f;
    }
    for (int tile = 0; tile < tiles; ++tile) {
        const int col = tile * 512 + int(lane) * 16;
        ushort4 packed[4];
        float scale[4], offset[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            packed[r] = *reinterpret_cast<device const ushort4 *>(w + (row + r) * packed_stride + col / 4);
            const int gi = (row + r) * group_stride + col / 64;
            scale[r] = block_metadata(scales[gi]);
            offset[r] = block_metadata(bias[gi]);
        }
        #pragma unroll
        for (uint b = 0; b < B; ++b) {
            device const float *xb = x + int(b) * cols + col;
            const float4 a = *reinterpret_cast<device const float4 *>(xb);
            const float4 c1 = *reinterpret_cast<device const float4 *>(xb + 4);
            const float4 c2 = *reinterpret_cast<device const float4 *>(xb + 8);
            const float4 d = *reinterpret_cast<device const float4 *>(xb + 12);
            const float4 combined = a + c1 + c2 + d;
            const float xsum = combined.x + combined.y + combined.z + combined.w;
            const float4 factors = float4(1.0f,0x1p-4f,0x1p-8f,0x1p-12f);
            const float4 xa = a * factors, xc1 = c1 * factors, xc2 = c2 * factors, xd = d * factors;
            #pragma unroll
            for (uint r = 0; r < 4; ++r) {
                totals[b][r] += fma(scale[r], block_u16_dot16(packed[r],xa,xc1,xc2,xd), offset[r] * xsum);
            }
        }
    }
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const float total = simd_sum(totals[b][r]);
            if (lane == 0) y[b * uint(rows) + uint(row) + r] = total;
        }
    }
}

template<uint B>
inline void block_dense(device const half *w, device const float *x,
                        device float *y, constant uint *p, uint gid, ushort lane) {
    const uint first_row = (gid / 32) * 4;
    const uint rows = p[0], cols = p[1];
    if (first_row >= rows) return;
    float totals[B][4];
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) totals[b][r] = 0.0f;
    }
    const uint column_tiles = cols / 32 + uint(cols % 32 != 0);
    for (uint tile = 0; tile < column_tiles; ++tile) {
        const uint c = tile * 32 + uint(lane);
        if (c >= cols) continue;
        float weights[4];
        #pragma unroll
        for (uint r = 0; r < 4; ++r)
            weights[r] = first_row + r < rows ? float(w[(first_row + r) * cols + c]) : 0.0f;
        #pragma unroll
        for (uint b = 0; b < B; ++b) {
            const float value = x[b * cols + c];
            #pragma unroll
            for (uint r = 0; r < 4; ++r) totals[b][r] += weights[r] * value;
        }
    }
    #pragma unroll
    for (uint b = 0; b < B; ++b) {
        #pragma unroll
        for (uint r = 0; r < 4; ++r) {
            const float total = simd_sum(totals[b][r]);
            if (lane == 0 && first_row + r < rows) y[b * rows + first_row + r] = total;
        }
    }
}

#define BLOCK_AFFINE(NAME,B,BITS,GROUP,M) \
kernel void NAME(device const uint *w [[buffer(0)]], device const M *s [[buffer(1)]], \
                 device const M *bias [[buffer(2)]], device const float *x [[buffer(3)]], \
                 device float *y [[buffer(4)]], constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    block_affine<B,BITS,GROUP,M>(w,s,bias,x,y,p,gid,lane); \
}
#define BLOCK_FORMAT(BITS,GROUP,B) \
BLOCK_AFFINE(matmul_q##BITS##_g##GROUP##_b##B,B,BITS,GROUP,float) \
BLOCK_AFFINE(matmul_q##BITS##_g##GROUP##_b##B##_bf16,B,BITS,GROUP,ushort)
#define BLOCK_BATCHES(BITS,GROUP) \
BLOCK_FORMAT(BITS,GROUP,1) BLOCK_FORMAT(BITS,GROUP,2) \
BLOCK_FORMAT(BITS,GROUP,3) BLOCK_FORMAT(BITS,GROUP,4)
BLOCK_BATCHES(4,32)
BLOCK_BATCHES(4,64)
BLOCK_BATCHES(4,128)
BLOCK_BATCHES(8,32)
BLOCK_BATCHES(8,64)
BLOCK_BATCHES(8,128)
#undef BLOCK_BATCHES
#undef BLOCK_FORMAT
#undef BLOCK_AFFINE

#define BLOCK_ALIGNED(NAME,B,M) \
kernel void NAME(device const ushort *w [[buffer(0)]], device const M *s [[buffer(1)]], \
                 device const M *bias [[buffer(2)]], device const float *x [[buffer(3)]], \
                 device float *y [[buffer(4)]], constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    block_aligned_q4<B,M>(w,s,bias,x,y,p,gid,lane); \
}
#define BLOCK_ALIGNED_BATCH(B) \
BLOCK_ALIGNED(matmul_q4_g64_b##B##_aligned,B,float) \
BLOCK_ALIGNED(matmul_q4_g64_b##B##_aligned_bf16,B,ushort)
BLOCK_ALIGNED_BATCH(1)
BLOCK_ALIGNED_BATCH(2)
BLOCK_ALIGNED_BATCH(3)
BLOCK_ALIGNED_BATCH(4)
#undef BLOCK_ALIGNED_BATCH
#undef BLOCK_ALIGNED

#define BLOCK_DENSE(B) \
kernel void matmul_f16_b##B(device const half *w [[buffer(0)]], device const float *x [[buffer(1)]], \
                 device float *y [[buffer(2)]], constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) { \
    block_dense<B>(w,x,y,p,gid,lane); \
}
BLOCK_DENSE(1)
BLOCK_DENSE(2)
BLOCK_DENSE(3)
BLOCK_DENSE(4)
#undef BLOCK_DENSE

kernel void copy_f32(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                     constant uint *p [[buffer(15)]], uint gid [[thread_position_in_grid]]) {
    if (gid < p[0]) y[gid] = x[gid];
}
