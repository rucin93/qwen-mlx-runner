// Standalone TensorOps experiment; this file is not in the production library.
// Requires Metal 4 and macOS 26.3+ for cooperative matmul input tensors.
// ABI: packed row-major Q4 words, lossless BF16 scale/bias metadata,
// token-major FP32 X/Y, p = {rows, cols, batch, padded_cols}; cols must be
// divisible by 64. padded_cols = round_up(cols, 128); padded X has 8 rows.
// Run tensor_pad_input first in the candidate command buffer. Its entire
// buffer(5) scratch tensor is initialized, including K and batch tile padding.
// One SIMD group (32 threads) computes M output rows for batch <= 8.
// Dispatch ceil(rows / M) threadgroups with exactly 32 threads per group.
// Input/dequantization/output stay FP32. TensorOps changes reduction order;
// explicit per-weight dequantization also changes the scalar grouping order.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

inline float tensor_q4_metadata(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void tensor_pad_input(device const float *x [[buffer(3)]],
                             device float *padded [[buffer(5)]],
                             constant uint *p [[buffer(15)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint cols = p[1], batch = p[2], padded_cols = p[3];
    if (gid >= 8u * padded_cols) return;
    const uint b = gid / padded_cols;
    const uint col = gid % padded_cols;
    padded[gid] = b < batch && col < cols ? x[b * cols + col] : 0.0f;
}

template<int M, int K>
inline void tensor_q4_g64(device const uint *w,
                          device const ushort *scales,
                          device const ushort *bias,
                          device float *padded,
                          device float *y,
                          constant uint *p,
                          uint group) {
    constexpr int N = 8;
    constexpr auto descriptor = matmul2d_descriptor(
        M, N, K, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> op;

    const uint rows = p[0], cols = p[1], batch = p[2], padded_cols = p[3];
    // The pointer is read-only in this shader. Its tensor element type must be
    // unqualified float for the TensorOps operand-type dispatch. The innermost
    // tensor stride must be one: preserve padded X [k, batch position]
    // and let the descriptor transpose the right operand for multiplication.
    auto activations = tensor(padded, dextents<int, 2>(int(padded_cols), N),
                              array<int, 2>{1, int(padded_cols)});
    auto initial_right = activations.slice(0, 0);
    auto left = op.template get_left_input_cooperative_tensor<float, float, float>();
    auto result = op.template get_destination_cooperative_tensor<
        decltype(left), decltype(initial_right), float>();

    // Construction allocates storage but does not establish accumulator values.
    for (ushort i = 0; i < result.get_capacity(); ++i) {
        result.set(i, 0.0f);
    }

    const uint first_row = group * uint(M);
    const uint words_per_row = cols / 8;
    const uint groups_per_row = cols / 64;

    for (uint first_col = 0; first_col < cols; first_col += uint(K)) {
        // Tensor coordinates use contiguous-column-first order: left [k, row].
        // Fill the cooperative layout through its public coordinate API so no
        // knowledge of the implementation's per-lane storage layout is needed.
        for (ushort i = 0; i < left.get_capacity(); ++i) {
            if (!left.is_valid_element(i)) continue;
            const auto coord = left.get_multidimensional_index(i);
            const uint col = first_col + uint(coord[0]);
            const uint row = first_row + uint(coord[1]);
            float value = 0.0f;
            if (row < rows && col < cols) {
                const uint packed = w[row * words_per_row + col / 8];
                const uint q = (packed >> ((col % 8) * 4)) & 15u;
                const uint gi = row * groups_per_row + col / 64;
                value = fma(float(q), tensor_q4_metadata(scales[gi]),
                            tensor_q4_metadata(bias[gi]));
            }
            left[i] = value;
        }

        // The padding kernel supplies explicit zero for unused N=8 positions
        // and the last K tile; this read always covers a fully allocated tile.
        // Only the dequantized weights are cooperative:
        // the macOS 26 implementation restricts the both-cooperative case to
        // N/K tile sizes 16 or 32.
        auto right = activations.slice(int(first_col), 0);
        op.run(left, right, result);
    }

    // The result coordinates are [batch position, row], whereas token-major Y
    // has the row axis contiguous. Store via coordinates rather than constructing
    // an inline tensor whose innermost stride would not be one.
    for (ushort i = 0; i < result.get_capacity(); ++i) {
        if (!result.is_valid_element(i)) continue;
        const auto coord = result.get_multidimensional_index(i);
        const uint b = uint(coord[0]);
        const uint row = first_row + uint(coord[1]);
        if (b < batch && row < rows) y[b * rows + row] = result[i];
    }
}

#define TENSOR_Q4_KERNEL(NAME, M, K) \
kernel void NAME(device const uint *w [[buffer(0)]], \
                 device const ushort *scales [[buffer(1)]], \
                 device const ushort *bias [[buffer(2)]], \
                 device float *y [[buffer(4)]], \
                 device float *padded [[buffer(5)]], \
                 constant uint *p [[buffer(15)]], \
                 uint group [[threadgroup_position_in_grid]]) { \
    tensor_q4_g64<M, K>(w, scales, bias, padded, y, p, group); \
}

TENSOR_Q4_KERNEL(tensor_q4_m16_k64, 16, 64)
TENSOR_Q4_KERNEL(tensor_q4_m16_k128, 16, 128)
TENSOR_Q4_KERNEL(tensor_q4_m32_k64, 32, 64)
TENSOR_Q4_KERNEL(tensor_q4_m32_k128, 32, 128)
