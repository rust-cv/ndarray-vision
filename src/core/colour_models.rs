/// Enum containing supported pixel formats for images. Storage type is
/// determined by the Image container
/// TODO consider representing this with a struct tag instead. This would allow
/// library users add other models more easily
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum ColourModel {
    /// Single channel intensity image
    Gray,
    /// Red Green Blue image
    RGB,
    /// RGB image with an added alpha channel
    RGBA,
    /// Hue Saturation Value image
    HSV,
    /// Hue Saturation Intensity image
    HSI,
    /// Hue Saturation Lightness image
    HSL,
    /// Y Chroma-red Chroma-blue image
    YCrCb,
    /// CIE 1931 X Y Z colour space - older standard meant to represent colours
    /// visible to the human eye
    CIEXYZ,
    /// CIE 1976 L*a*b* colour space - recommended for characterisation of
    /// coloured surfaces and dyes
    CIELAB,
    /// CIE 1976 L* U* V* colour space - recommended for characterisation of
    /// colour displays
    CIELUV,
    /// Unspecified layout with the given number of channels
    Other(usize),
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum ColourError {
    InvalidDataDimensions,
}

impl ColourModel {
    /// Returns the number of channels used to represent the colour
    pub fn channels(&self) -> usize {
        use ColourModel::*;
        match self {
            Gray => 1,
            RGB | HSV | HSI | HSL | YCrCb | CIELAB | CIEXYZ | CIELUV => 3,
            RGBA => 4,
            Other(n) => *n,
        }
    }
}
