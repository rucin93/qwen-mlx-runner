# Specialized affine matrix-vector kernels

`kernels/matvec_fast.metal` and `kernels/matvec_aligned.metal` are original implementations for the existing
row-major, little-endian packed affine weights. It does not change the checkpoint
format or use an inference library. The generic kernel in `kernels/qwen.metal`
remains the reference and fallback.

## Dispatch contract

| Kernel family | Bits | Group sizes | Rows per SIMD | Logical grid threads |
| --- | --- | --- | --- | --- |
| `matvec_q4_g{32,64,128}` | 4 | 32, 64, 128 | 4 | `ceil(rows / 4) * 32` |
| `matvec_q8_g{32,64,128}` | 8 | 32, 64, 128 | 4 | `ceil(rows / 4) * 32` |
| `matvec_q4_g{32,64,128}_aligned` | 4 | 32, 64, 128 | 4 | `(rows / 4) * 32` |

Use 128 threads per threadgroup for the general packed kernels and 64 for the
aligned kernels. Rounding the grid
up to a full threadgroup is safe. A main-kernel threadgroup produces up to 16
output rows. Each SIMD processes four rows and shares its input-vector loads
across those rows. No threadgroup memory or barriers are required.

Buffers are unchanged: packed `uint` weights at 0, FP32 scales at 1, FP32 affine
biases at 2, FP32 input vector at 3, FP32 output at 4. Buffer 15 contains four
`uint` parameters: rows, columns, bits, and group size. Bits and group size are
compile-time constants in these kernels; the host must select the matching
specialization. Columns must be divisible by the selected group size. Arbitrary
row counts, including a final incomplete group of four rows, are supported by
the general packed kernels. Aligned kernels require rows divisible by four,
columns divisible by 512 and `rows * cols <= INT_MAX`; the host checks these
conditions and otherwise selects the general kernel. Unsupported group sizes
continue to use the original generic shader.

## Work reduction

The reference loops over individual weights and computes the packed-word and
quantization-group indices using runtime division. The specialization assigns
one full packed word to each SIMD lane, making all packing and group arithmetic
compile-time powers of two. Each lane loads its word once, unpacks its eight
4-bit or four 8-bit values, and evaluates a vector dot product. Adjacent lanes
load adjacent words from each row. The main variant reuses each input vector
slice across four independent output rows.

The aligned variant processes 16 Q4 values per lane with vectorized `ushort4`
loads and signed offsets. Four explicit row accumulators limit register
lifetime. Power-of-two input scaling lets masked 16-bit fields participate
directly in the dot product, avoiding shifts for each nibble. There are no
per-row or per-column tail checks inside this kernel's main loop. Input scaling
can flush extreme subnormal values; this is an FP32 fast-math inference path,
not a bitwise IEEE arithmetic oracle. Reference kernels retain strict math.

The affine evaluation is factored over each packed word:

```
sum_i ((q_i * scale + bias) * x_i)
    = scale * sum_i(q_i * x_i) + bias * sum_i(x_i)
```

This reduces affine arithmetic and shares the input sum across rows. The final
sum is reduced using `simd_sum`. It changes FP32 evaluation and accumulation
order, so results need tolerance-based comparison, not bitwise comparison.
In particular, near-zero output values require an absolute tolerance as well
as a relative tolerance. Large bias/scale cancellation should be included in
numerical tests. The CPU reference should apply affine dequantization to each
weight before accumulation to avoid duplicating the optimization under test.

## Evidence boundary

Kernel-level GPU timings and correctness results belong to the host benchmark
and test logs. This design alone establishes no throughput claim, and a kernel
speedup does not establish an equal end-to-end token-generation speedup.
The target M5 still requires validation; the selected default is based on the
available M1 measurements. Tests cover both quantization widths, all supported
fast group sizes, partial final row groups, and columns below or not divisible
by a SIMD's full packed-word span.

The optimized host path uses static pipeline names and a stack-backed list of
shader outputs for the barrier API. The engine uses a serial compute encoder;
Metal ignores explicit memory barriers on that encoder, so this change is a
host-allocation reduction, not evidence of reduced GPU fencing.
