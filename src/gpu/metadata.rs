//! Lossless storage conversion for affine quantization metadata.

/// Keep the high 16 bits only when they reconstruct every finite FP32 value
/// exactly. This preserves signed zeros and BF16 subnormals without converting
/// through F16 or rounding an arbitrary FP32 value.
pub fn pack_bf16_exact(values: &[f32]) -> Option<Vec<u16>> {
    if !values
        .iter()
        .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0)
    {
        return None;
    }
    Some(values.iter().map(|v| (v.to_bits() >> 16) as u16).collect())
}

pub(super) fn supported_bf16_quantization(bits: u32, group: usize) -> bool {
    matches!(bits, 4 | 8) && matches!(group, 32 | 64 | 128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_packing_preserves_signed_zero_and_tiny_values() {
        let patterns = [
            0x0000_0000, // +0
            0x8000_0000, // -0
            0x3fc0_0000, // 1.5
            0xc020_0000, // -2.5
            0x0001_0000, // Smallest positive BF16 subnormal.
            0x8001_0000,
            0x0080_0000, // Smallest normal FP32/BF16 value.
            0x7f7f_0000, // Largest finite BF16 value; much larger than F16.
        ];
        let values: Vec<_> = patterns.into_iter().map(f32::from_bits).collect();
        let packed = pack_bf16_exact(&values).unwrap();
        assert_eq!(
            packed
                .into_iter()
                .map(|v| u32::from(v) << 16)
                .collect::<Vec<_>>(),
            patterns
        );
    }

    #[test]
    fn inexact_or_nonfinite_metadata_is_not_compacted() {
        for value in [
            0.1,
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            assert!(pack_bf16_exact(&[1.0, value, -0.0]).is_none());
        }
    }

    #[test]
    fn compact_metadata_has_explicit_kernel_coverage() {
        for bits in [4, 8] {
            for group in [32, 64, 128] {
                assert!(supported_bf16_quantization(bits, group));
            }
        }
        for (bits, group) in [(2, 64), (16, 64), (4, 8), (4, 256), (8, 0)] {
            assert!(!supported_bf16_quantization(bits, group));
        }
    }
}
