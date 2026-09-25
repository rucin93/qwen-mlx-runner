// Experimental original Q4 affine GEMV: one quantization group per lane.
// ABI: buffers 0..4 = packed Q4 weights, FP32 scales, FP32 affine biases,
// FP32 input, FP32 output. Buffer 15 = uint rows, cols, bits, group_size.
// Dispatch ceil(rows / 4) * 32 logical threads, 64 threads per threadgroup.
// Each SIMD has four independent 8-lane row reductions. Padded rows and a
// final incomplete set of eight quantization groups are guarded explicitly.
// Host requirements: cols % GROUP == 0 and signed-int-safe index products.
// Configure pipeline maxTotalThreadsPerThreadgroup = 64, SIMD width = 32.
//
// Compared with aligned at GROUP=64, a SIMD processes the same four rows and
// 512 columns per outer iteration. Each lane streams 64 weights from one row
// instead of 16 weights from four rows, so there is one metadata load pair and
// affine application per lane rather than four. Only one output accumulator
// is live. This trades input reuse between rows for simpler metadata work;
// the extra input reads should hit cache, but performance must be measured.
// FP32 arithmetic throughout. As in aligned, exact power-of-two prescaling
// can flush extreme subnormal inputs under fast math.
#include <metal_stdlib>
using namespace metal;

inline float stream_u16_dot4(ushort q, float4 x) {
    return dot(float4(ushort4(q & ushort(0x000f), q & ushort(0x00f0),
                             q & ushort(0x0f00), q & ushort(0xf000))), x);
}

template<int GROUP>
inline void affine_q4_stream(device const ushort *w,
                             device const float *scales,
                             device const float *bias,
                             device const float *x,
                             device float *y,
                             constant uint *p,
                             uint gid, ushort lane) {
    if (p[2] != 4 || p[3] != uint(GROUP)) return;
    const int first_row = (int(gid) / 32) * 4;
    if (first_row >= int(p[0])) return;
    const int row = first_row + int(lane) / 8;
    const int lane_in_row = int(lane) & 7;
    const int cols = int(p[1]);
    const int groups_per_row = cols / GROUP;
    const int packed_stride = cols / 4;
    const bool valid_row = row < int(p[0]);
    const int weight_base = valid_row ? row * packed_stride : 0;
    const int scale_base = valid_row ? row * groups_per_row : 0;
    float total = 0.0f;
    for (int group_tile = 0; group_tile < groups_per_row; group_tile += 8) {
        const int group = group_tile + lane_in_row;
        if (valid_row && group < groups_per_row) {
            const int col = group * GROUP;
            float qdot = 0.0f;
            float xsum = 0.0f;
            // Keep one 16-element input slice live at a time. The quant group
            // dot and input sum stay in FP32; no conversion of x to FP16.
            #pragma unroll
            for (int offset = 0; offset < GROUP; offset += 16) {
                const float4 a = *reinterpret_cast<device const float4 *>(x + col + offset);
                const float4 b = *reinterpret_cast<device const float4 *>(x + col + offset + 4);
                const float4 c = *reinterpret_cast<device const float4 *>(x + col + offset + 8);
                const float4 d = *reinterpret_cast<device const float4 *>(x + col + offset + 12);
                const ushort4 q = *reinterpret_cast<device const ushort4 *>(
                    w + weight_base + (col + offset) / 4);
                const float4 combined = a + b + c + d;
                xsum += combined.x + combined.y + combined.z + combined.w;
                const float4 factors = float4(1.0f, 0x1p-4f, 0x1p-8f, 0x1p-12f);
                qdot += stream_u16_dot4(q.x, a * factors)
                      + stream_u16_dot4(q.y, b * factors)
                      + stream_u16_dot4(q.z, c * factors)
                      + stream_u16_dot4(q.w, d * factors);
            }
            const int si = scale_base + group;
            total += fma(scales[si], qdot, bias[si] * xsum);
        }
    }
    // XOR 1, 2, 4 never crosses an 8-lane row subgroup. All lanes, including
    // padded rows/groups, reach these shuffles; padding contributed zero.
    total += simd_shuffle_xor(total, ushort(1));
    total += simd_shuffle_xor(total, ushort(2));
    total += simd_shuffle_xor(total, ushort(4));
    if (lane_in_row == 0 && row < int(p[0])) y[row] = total;
}

#define AFFINE_STREAM_ENTRY(NAME, GROUP) \
kernel void NAME(device const ushort *w [[buffer(0)]], \
                 device const float *scales [[buffer(1)]], \
                 device const float *bias [[buffer(2)]], \
                 device const float *x [[buffer(3)]], \
                 device float *y [[buffer(4)]], \
                 constant uint *p [[buffer(15)]], \
                 uint gid [[thread_position_in_grid]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    affine_q4_stream<GROUP>(w,scales,bias,x,y,p,gid,lane); \
}

AFFINE_STREAM_ENTRY(matvec_q4_g32_stream,32)
AFFINE_STREAM_ENTRY(matvec_q4_g64_stream,64)
AFFINE_STREAM_ENTRY(matvec_q4_g128_stream,128)

#undef AFFINE_STREAM_ENTRY
