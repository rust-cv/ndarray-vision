/// When working with pixel data types may have odd bitdepths or not use the
/// full range of the value. We can't assume every image with `u8` ranges from
/// [0..255]. Additionally, floating point representations of pixels normally
/// range from [0.0..1.0]. `PixelBound` is an attempt to solve this issue.
///
/// Unfortunately, type aliases don't really create new types so if you wanted
/// to create a pixel with a reduced bound you'd have to create something like:
///
/// ```Rust
/// struct LimitedU8(u8);
/// impl PixelBound for LimitedU8 {
///     fn min_pixel() -> Self {
///         LimitedU8(16u8)
///     }
///     
///     fn max_pixel() -> Self {
///         LimitedU8(160u8)
///     }
/// }
///
/// And then implement the required numerical traits just calling the
/// corresponding methods in `u8`
/// ```
pub trait PixelBound {
    /// The minimum value a pixel can take
    fn min_pixel() -> Self;
    /// The maximum value a pixel can take
    fn max_pixel() -> Self;
    /// If this is a non-floating point value return true
    fn is_integral() -> bool {
        true
    }
}

impl PixelBound for f64 {
    fn min_pixel() -> Self {
        0.0f64
    }

    fn max_pixel() -> Self {
        1.0f64
    }

    fn is_integral() -> bool {
        false
    }
}

impl PixelBound for f32 {
    fn min_pixel() -> Self {
        0.0f32
    }

    fn max_pixel() -> Self {
        1.0f32
    }

    fn is_integral() -> bool {
        false
    }
}

impl PixelBound for u8 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for u16 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for u32 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for u64 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for u128 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for i8 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for i16 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for i32 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for i64 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

impl PixelBound for i128 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_correct() {
        assert!(!f64::is_integral());
        assert!(!f32::is_integral());
        assert!(u128::is_integral());
        assert!(u64::is_integral());
        assert!(u32::is_integral());
        assert!(u16::is_integral());
        assert!(u8::is_integral());
        assert!(i128::is_integral());
        assert!(i64::is_integral());
        assert!(i32::is_integral());
        assert!(i16::is_integral());
        assert!(i8::is_integral());
    }

    #[test]
    fn max_more_than_min() {
        assert!(f64::max_pixel() > f64::min_pixel());
        assert!(f32::max_pixel() > f32::min_pixel());
        assert!(u8::max_pixel() > u8::min_pixel());
        assert!(u16::max_pixel() > u16::min_pixel());
        assert!(u32::max_pixel() > u32::min_pixel());
        assert!(u64::max_pixel() > u64::min_pixel());
        assert!(u128::max_pixel() > u128::min_pixel());
        assert!(i8::max_pixel() > i8::min_pixel());
        assert!(i16::max_pixel() > i16::min_pixel());
        assert!(i32::max_pixel() > i32::min_pixel());
        assert!(i64::max_pixel() > i64::min_pixel());
        assert!(i128::max_pixel() > i128::min_pixel());
    }
}
