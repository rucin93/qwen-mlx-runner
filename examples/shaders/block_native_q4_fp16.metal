// Standalone native signed-int4 TensorOps variants. Concatenated after
// block_affine_fp16.metal, whose preparation and BF16 helper they share.
// Requires Metal 4 with macOS 26.4+ packed integer tensor support.
//
// Buffer 8 contains a one-time exact re-encoding of the original Q4 words:
// original_word ^ 0x88888888u. Each signed nibble is therefore q-8.
// Its bound offset is 128-byte aligned, physical row stride is a multiple of
// 256 nibbles (128 bytes), and physical rows are rounded up to 16. Fill all
// padding words with zero (signed zero). No FP16 weight conversion is involved.
// p[4] supplies the physical signed-weight stride in nibbles. p[0..3] retain
// {rows, cols, batch, padded_cols}; scales/bias retain the original row stride.
// Both variants dispatch ceil(rows/16) groups of 32 threads. The packed variant
// uses one MMA for all high/low channels and merges in 512 bytes of TG scratch.
#include <metal_stdlib>
#include <metal_tensor>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

#if !defined(__HAVE_INT4B_FORMAT_TYPE__) || !__HAVE_INT4B_FORMAT_TYPE__
#error "Native Q4 TensorOps requires packed int4 tensor support (macOS 26.4+)"
#endif

using namespace metal;
using namespace mpp::tensor_ops;

template<bool Packed>
inline void affine_native_q4_m16(device uint *signed_weights,
                                 device const ushort *scales,
                                 device const ushort *bias,
                                 device half *high,
                                 device half *packed,
                                 device const float *metadata,
                                 device float *y,
                                 constant uint *p,
                                 threadgroup float *partials,
                                 uint group,
                                 ushort lane) {
    constexpr int Tokens = 8, Rows = 16, Group = 64;
    constexpr auto descriptor = matmul2d_descriptor(
        Tokens, Rows, Group, false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroup> op;

    const uint rows = p[0], cols = p[1], batch = p[2], padded_cols = p[3];
    const uint signed_stride = p[4];
    const uint physical_rows = (rows + 15u) / 16u * 16u;
    const uint padded_groups = padded_cols / 64;
    const uint actual_groups = cols / 64;
    const uint first_row = group * 16;
    device half *input = Packed ? packed : high;
    auto activations = tensor(input, dextents<int, 2>(int(padded_cols), Tokens),
                              array<int, 2>{1, int(padded_cols)});
    tensor<device int4b_format, dextents<int, 2>, tensor_inline> weights(
        reinterpret_cast<device uchar *>(signed_weights),
        dextents<int, 2>(int(signed_stride), int(physical_rows)),
        array<int, 2>{1, int(signed_stride)});
    auto initial_left = activations.slice(0, 0);
    auto initial_right = weights.slice(0, int(first_row));
    auto dot = op.template get_destination_cooperative_tensor<
        decltype(initial_left), decltype(initial_right), float>();
    auto total = op.template get_destination_cooperative_tensor<
        decltype(initial_left), decltype(initial_right), float>();
    for (ushort i = 0; i < total.get_capacity(); ++i) total.set(i, 0.0f);

    for (uint g = 0; g < actual_groups; ++g) {
        auto left = activations.slice(int(g * 64), 0);
        auto right = weights.slice(int(g * 64), int(first_row));
        op.run(left, right, dot);
        for (ushort i = 0; i < total.get_capacity(); ++i) {
            if (!total.is_valid_element(i)) continue;
            const auto coord = total.get_multidimensional_index(i);
            const uint row = first_row + uint(coord[0]);
            const uint channel = uint(coord[1]);
            const uint b = Packed ? channel % 4 : channel;
            if (row >= rows || b >= batch) continue;
            const uint mi = (b * padded_groups + g) * 2;
            const float alpha = metadata[mi];
            const uint wi = row * actual_groups + g;
            const float scale = affine_fp16_bf16(scales[wi]);
            if (!Packed || channel < 4) {
                const float offset = fma(8.0f, scale, affine_fp16_bf16(bias[wi]));
                total[i] = fma(offset, metadata[mi + 1], total[i]);
            }
            total[i] = fma(scale * alpha, dot[i], total[i]);
        }
    }

    if (Packed) {
        // Keep low-channel totals scaled by 2048 until the final merge. Every
        // physical tile slot was initialized, including inactive token/row tails.
        for (ushort i = 0; i < total.get_capacity(); ++i) {
            if (!total.is_valid_element(i)) continue;
            const auto coord = total.get_multidimensional_index(i);
            partials[uint(coord[1]) * 16 + uint(coord[0])] = total[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = uint(lane); i < batch * 16; i += 32) {
            const uint b = i / 16, local_row = i % 16;
            const uint row = first_row + local_row;
            if (row < rows) {
                y[b * rows + row] = fma(partials[(b + 4) * 16 + local_row],
                                        0x1p-11f, partials[b * 16 + local_row]);
            }
        }
    } else {
        for (ushort i = 0; i < total.get_capacity(); ++i) {
            if (!total.is_valid_element(i)) continue;
            const auto coord = total.get_multidimensional_index(i);
            const uint row = first_row + uint(coord[0]);
            const uint b = uint(coord[1]);
            if (row < rows && b < batch) y[b * rows + row] = total[i];
        }
    }
}

#define AFFINE_NATIVE_Q4_KERNEL(NAME, PACKED) \
kernel void NAME(device const ushort *scales [[buffer(1)]], \
                 device const ushort *bias [[buffer(2)]], \
                 device float *y [[buffer(4)]], \
                 device half *high [[buffer(5)]], \
                 device const float *metadata [[buffer(7)]], \
                 device uint *signed_weights [[buffer(8)]], \
                 device half *packed [[buffer(9)]], \
                 constant uint *p [[buffer(15)]], \
                 uint group [[threadgroup_position_in_grid]], \
                 ushort lane [[thread_index_in_simdgroup]]) { \
    threadgroup float partials[16 * 8]; \
    affine_native_q4_m16<PACKED>(signed_weights, scales, bias, high, packed, \
                                  metadata, y, p, partials, group, lane); \
}

AFFINE_NATIVE_Q4_KERNEL(affine_native_q4_m16_single, false)
AFFINE_NATIVE_Q4_KERNEL(affine_native_q4_m16_packed, true)
