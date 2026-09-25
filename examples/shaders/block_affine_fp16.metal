// Standalone group-affine FP16 experiment. No production dispatch uses this file.
// Q4 words and BF16 scale/bias bits remain unchanged. Centered Q4 integers are
// exact half values; only normalized activations are rounded to half.
// Metal 4, macOS 26.3+, fast math disabled.
//
// Buffers: 0=uint Q4, 1=ushort scales, 2=ushort bias, 3=original float X,
// 4=float Y, 5=half high X, 6=half scaled residual X, 7=float group metadata,
// 9=half combined X (high token channels 0..3, low token channels 4..7),
// 15=uint {rows, cols, batch, padded_cols}. cols and padded_cols are multiples
// of 64; padded_cols >= cols; batch <= 4. X/Y are token-major.
// High/low scratch each has 8*padded_cols half elements. Metadata contains
// 8*(padded_cols/64) pairs [alpha, original_float_sum], in token-major order.
// Prepare: 8*(padded_cols/64) threadgroups, exactly 32 threads each.
// Matmul: ceil(rows/M) threadgroups, exactly 32 threads each.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline float affine_fp16_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void affine_fp16_prepare(device const float *x [[buffer(3)]],
                                device half *high [[buffer(5)]],
                                device half *low [[buffer(6)]],
                                device float *metadata [[buffer(7)]],
                                device half *packed [[buffer(9)]],
                                constant uint *p [[buffer(15)]],
                                uint group [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    const uint cols = p[1], batch = p[2], padded_cols = p[3];
    const uint groups = padded_cols / 64;
    const uint b = group / groups;
    const uint g = group % groups;
    const uint c0 = g * 64 + uint(lane);
    const uint c1 = c0 + 32;
    const float x0 = b < batch && c0 < cols ? x[b * cols + c0] : 0.0f;
    const float x1 = b < batch && c1 < cols ? x[b * cols + c1] : 0.0f;
    const float maximum = simd_max(max(fabs(x0), fabs(x1)));

    // Floor power-of-two scaling puts the largest normal value in [1,2).
    // Avoid computing 1/alpha: it could overflow for tiny, finite activations.
    // Subnormal-only groups use the smallest normal float as their scale.
    const uint exponent_bits = as_type<uint>(maximum) & 0x7f800000u;
    const float alpha = maximum == 0.0f ? 1.0f
        : as_type<float>(max(exponent_bits, 0x00800000u));
    const float v0 = x0 / alpha;
    const float v1 = x1 / alpha;
    const half h0 = half(v0), h1 = half(v1);
    const half l0 = half((v0 - float(h0)) * 2048.0f);
    const half l1 = half((v1 - float(h1)) * 2048.0f);
    high[b * padded_cols + c0] = h0;
    high[b * padded_cols + c1] = h1;
    low[b * padded_cols + c0] = l0;
    low[b * padded_cols + c1] = l1;
    // Every packed channel is written exactly once. Preparation groups b>=4
    // still initialize ordinary high/low scratch, but must not overwrite the
    // low channels written by groups b<4.
    if (b < 4) {
        packed[b * padded_cols + c0] = h0;
        packed[b * padded_cols + c1] = h1;
        packed[(b + 4) * padded_cols + c0] = l0;
        packed[(b + 4) * padded_cols + c1] = l1;
    }

    // The affine offset multiplies the original FP32 activation sum, never
    // the rounded high/low reconstruction. Fixtures must have finite sums.
    const float original_sum = simd_sum(x0 + x1);
    if (lane == 0) {
        metadata[group * 2] = alpha;
        metadata[group * 2 + 1] = original_sum;
    }
}

template<int M, bool Compensated>
inline void affine_fp16_group_matmul(device const uint *w,
                                     device const ushort *scales,
                                     device const ushort *bias,
                                     device half *high,
                                     device half *low,
                                     device const float *metadata,
                                     device float *y,
                                     constant uint *p,
                                     uint group) {
    constexpr int N = 8, K = 64;
    constexpr auto descriptor = matmul2d_descriptor(
        M, N, K, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroup> op;

    const uint rows = p[0], cols = p[1], batch = p[2], padded_cols = p[3];
    const uint padded_groups = padded_cols / 64;
    const uint actual_groups = cols / 64;
    const uint first_row = group * uint(M);
    const uint words_per_row = cols / 8;
    auto high_tensor = tensor(high, dextents<int, 2>(int(padded_cols), N),
                              array<int, 2>{1, int(padded_cols)});
    auto low_tensor = tensor(low, dextents<int, 2>(int(padded_cols), N),
                             array<int, 2>{1, int(padded_cols)});
    auto initial_right = high_tensor.slice(0, 0);
    auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto high_dot = op.template get_destination_cooperative_tensor<
        decltype(left), decltype(initial_right), float>();
    auto low_dot = op.template get_destination_cooperative_tensor<
        decltype(left), decltype(initial_right), float>();
    auto total = op.template get_destination_cooperative_tensor<
        decltype(left), decltype(initial_right), float>();
    for (ushort i = 0; i < total.get_capacity(); ++i) total.set(i, 0.0f);

    for (uint g = 0; g < actual_groups; ++g) {
        // q-8 is in [-8,7], exactly representable in half. Scale/bias remain
        // FP32 and are applied after each group-specific matrix product.
        for (ushort i = 0; i < left.get_capacity(); ++i) {
            if (!left.is_valid_element(i)) continue;
            const auto coord = left.get_multidimensional_index(i);
            const uint col = g * 64 + uint(coord[0]);
            const uint row = first_row + uint(coord[1]);
            half value = half(0.0f);
            if (row < rows) {
                const uint packed = w[row * words_per_row + col / 8];
                const int q = int((packed >> ((col % 8) * 4)) & 15u);
                value = half(q - 8);
            }
            left[i] = value;
        }
        auto high_tile = high_tensor.slice(int(g * 64), 0);
        op.run(left, high_tile, high_dot);
        if (Compensated) {
            auto low_tile = low_tensor.slice(int(g * 64), 0);
            op.run(left, low_tile, low_dot);
        }

        for (ushort i = 0; i < total.get_capacity(); ++i) {
            if (!total.is_valid_element(i)) continue;
            const auto coord = total.get_multidimensional_index(i);
            const uint b = uint(coord[0]);
            const uint row = first_row + uint(coord[1]);
            if (b >= batch || row >= rows) continue;
            const uint mi = (b * padded_groups + g) * 2;
            const float alpha = metadata[mi];
            const float original_sum = metadata[mi + 1];
            const uint wi = row * actual_groups + g;
            const float scale = affine_fp16_bf16(scales[wi]);
            const float offset = fma(8.0f, scale, affine_fp16_bf16(bias[wi]));
            float dot = high_dot[i];
            if (Compensated) dot = fma(low_dot[i], 0x1p-11f, dot);
            total[i] = fma(offset, original_sum, total[i]);
            total[i] = fma(scale * alpha, dot, total[i]);
        }
    }

    for (ushort i = 0; i < total.get_capacity(); ++i) {
        if (!total.is_valid_element(i)) continue;
        const auto coord = total.get_multidimensional_index(i);
        const uint b = uint(coord[0]);
        const uint row = first_row + uint(coord[1]);
        if (b < batch && row < rows) y[b * rows + row] = total[i];
    }
}

#define AFFINE_FP16_KERNEL(NAME, M, COMPENSATED) \
kernel void NAME(device const uint *w [[buffer(0)]], \
                 device const ushort *scales [[buffer(1)]], \
                 device const ushort *bias [[buffer(2)]], \
                 device float *y [[buffer(4)]], \
                 device half *high [[buffer(5)]], \
                 device half *low [[buffer(6)]], \
                 device const float *metadata [[buffer(7)]], \
                 constant uint *p [[buffer(15)]], \
                 uint group [[threadgroup_position_in_grid]]) { \
    affine_fp16_group_matmul<M, COMPENSATED>( \
        w, scales, bias, high, low, metadata, y, p, group); \
}

AFFINE_FP16_KERNEL(affine_fp16_m16_single, 16, false)
AFFINE_FP16_KERNEL(affine_fp16_m16_compensated, 16, true)
