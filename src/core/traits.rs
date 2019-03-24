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

    /// Returns the number of discrete levels. This is an option because it's 
    /// deemed meaningless for types like floats which suffer from rounding 
    /// issues
    fn discrete_levels() -> Option<usize>;
}

impl PixelBound for f64 {
    fn min_pixel() -> Self {
        0.0f64
    }

    fn max_pixel() -> Self {
        1.0f64
    }

    fn discrete_levels() -> Option<usize> {
        None
    }
}

impl PixelBound for f32 {
    fn min_pixel() -> Self {
        0.0f32
    }

    fn max_pixel() -> Self {
        1.0f32
    }

    fn discrete_levels() -> Option<usize> {
        None
    }
}

impl PixelBound for u8 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for u16 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for u32 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for u64 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for u128 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for i8 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for i16 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for i32 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for i64 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}

impl PixelBound for i128 {
    fn min_pixel() -> Self {
        Self::min_value()
    }

    fn max_pixel() -> Self {
        Self::max_value()
    }

    fn discrete_levels() -> Option<usize> {
        Some(Self::max_value() as usize + 1)
    }
}
