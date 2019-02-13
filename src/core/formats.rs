

pub enum PixelFormat {
    Gray,
    RGB,
    RGBA,
    HSV,
    HSI,
    HSL,
    YCrCb,
    CIEXYZ,
    CIELAB,
    CIELUV,
    /// Unspecified layout with the given number of channels
    Other(usize),
}


impl PixelFormat {

    pub fn channels(&self) -> usize {
        use PixelFormat::*;
        match self {
            Gray => 1,
            RGB | HSV | HSI | HSL | YCrCb | CIELAB | CIEXYZ | CIELUV => 3,
            RGBA => 4,
            Other(n) => *n
        }
    }
}
