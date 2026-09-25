// Standalone compensated group-affine experiment. Concatenate after
// block_affine_fp16.metal for its public TensorOps imports and BF16 helper.
// No production dispatch uses this file.
//
// Buffers: 0=original uint Q4, 1=ushort scales, 2=ushort bias, 4=float Y,
// 7=group metadata [alpha, original FP32 sum], 9=packed half activations,
// 15=uint {rows, cols, batch, padded_cols, ...}. batch is 1..4.
// Prepare fills all 8*padded_cols values in buffer 9: channels 0..3 are high
// inputs, channels 4..7 are residuals multiplied by 2048. Inactive channels
// and column padding are zero. Metadata retains original token indexing.
// Dispatch ceil(rows/16) threadgroups, exactly 32 threads each.
// One M16/N8/K64 matrix operation computes both high and low terms per group.
// Their FP32 sums are combined through 512 bytes of threadgroup storage only
// after all groups, so this has a different reduction order from two-MMA mode.

kernel void affine_fp16_m16_packed(
    device const uint *w [[buffer(0)]],
    device const ushort *scales [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device float *y [[buffer(4)]],
    device const float *metadata [[buffer(7)]],
    device half *packed_x [[buffer(9)]],
    constant uint *p [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int M = 16, N = 8, K = 64;
    constexpr auto descriptor = matmul2d_descriptor(
        M, N, K, false, true, false, matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroup> op;
    threadgroup float channels[M * N];

    const uint rows = p[0], cols = p[1], batch = p[2], padded_cols = p[3];
    const uint padded_groups = padded_cols / 64;
    const uint actual_groups = cols / 64;
    const uint words_per_row = cols / 8;
    const uint first_row = group * uint(M);
    auto activations = tensor(packed_x, dextents<int, 2>(int(padded_cols), N),
                              array<int, 2>{1, int(padded_cols)});
    auto initial_right = activations.slice(0, 0);
    auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
    auto dot = op.template get_destination_cooperative_tensor<
        decltype(left), decltype(initial_right), float>();
    auto total = op.template get_destination_cooperative_tensor<
        decltype(left), decltype(initial_right), float>();
    for (ushort i = 0; i < total.get_capacity(); ++i) {
        if (total.is_valid_element(i)) total[i] = 0.0f;
    }

    for (uint g = 0; g < actual_groups; ++g) {
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
        auto right = activations.slice(int(g * 64), 0);
        op.run(left, right, dot);

        for (ushort i = 0; i < total.get_capacity(); ++i) {
            if (!total.is_valid_element(i)) continue;
            const auto coord = total.get_multidimensional_index(i);
            const uint channel = uint(coord[0]);
            const uint b = channel % 4;
            const uint row = first_row + uint(coord[1]);
            if (b >= batch || row >= rows) continue;
            const uint mi = (b * padded_groups + g) * 2;
            const uint wi = row * actual_groups + g;
            const float scale = affine_fp16_bf16(scales[wi]);
            const float coefficient = scale * metadata[mi];
            if (channel < 4) {
                const float offset = fma(8.0f, scale, affine_fp16_bf16(bias[wi]));
                total[i] = fma(offset, metadata[mi + 1], total[i]);
                total[i] = fma(coefficient, dot[i], total[i]);
            } else {
                total[i] = fma(coefficient * 0x1p-11f, dot[i], total[i]);
            }
        }
    }

    // Publish every channel, including explicit zeros for inactive tokens and
    // row tails. All threads reach the barrier, even in the last partial tile.
    for (ushort i = 0; i < total.get_capacity(); ++i) {
        if (!total.is_valid_element(i)) continue;
        const auto coord = total.get_multidimensional_index(i);
        channels[uint(coord[0]) * uint(M) + uint(coord[1])] = total[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = uint(lane); index < batch * uint(M); index += 32) {
        const uint b = index / uint(M);
        const uint local_row = index % uint(M);
        const uint row = first_row + local_row;
        if (row < rows) {
            y[b * rows + row] = channels[index] + channels[(b + 4) * uint(M) + local_row];
        }
    }
}
