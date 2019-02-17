use crate::core::Image;
use num_traits::{Num, NumAssignOps};
use num_traits::cast::{NumCast, FromPrimitive};
use ndarray::Array3;
use std::fmt::Display;
use std::convert::From;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Gray;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct RGB;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct RGBA;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct HSV;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct HSI;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct HSL;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct YCrCb;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct CIEXYZ;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct CIELAB;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct CIELUV;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Generic1;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Generic2;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Generic3;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Generic4;
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Generic5;

pub trait ColourModel {
    /// Number of colour channels for a type.
    fn channels() -> usize {
        3
    }
}

impl <T> From<Image<T, RGB>> for Image<T, HSV> where T: Copy + Clone + FromPrimitive + Num + NumAssignOps + NumCast + PartialOrd + Display {
    fn from(image: Image<T, RGB>) -> Self {
        let res = Array3::<T>::zeros((image.rows(), image.cols(), HSV::channels()));
        
        Self::from_data(res)
    }
}


impl ColourModel for RGB{}
impl ColourModel for HSV{}
impl ColourModel for HSI{}
impl ColourModel for HSL{}
impl ColourModel for YCrCb{}
impl ColourModel for CIEXYZ{}
impl ColourModel for CIELAB{}
impl ColourModel for CIELUV{}

impl ColourModel for Gray {
    fn channels() -> usize {
        1
    }
}

impl ColourModel for Generic1 {
    fn channels() -> usize {
        1
    }
}
impl ColourModel for Generic2 {
    fn channels() -> usize {
        2
    }
}
impl ColourModel for Generic3 {
    fn channels() -> usize {
        3
    }
}
impl ColourModel for Generic4 {
    fn channels() -> usize {
        4
    }
}
impl ColourModel for Generic5 {
    fn channels() -> usize {
        5
    }
}

impl ColourModel for RGBA {
    fn channels() -> usize {
        4
    }
}

/// Enum containing supported pixel formats for images. Storage type is
/// determined by the Image container
/// TODO consider representing this with a struct tag instead. This would allow
/// library users add other models more easily
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum ColourModels {
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

impl ColourModels {
    /// Returns the number of channels used to represent the colour
    pub fn channels(&self) -> usize {
        use ColourModels::*;
        match self {
            Gray => 1,
            RGB | HSV | HSI | HSL | YCrCb | CIELAB | CIEXYZ | CIELUV => 3,
            RGBA => 4,
            Other(n) => *n,
        }
    }
}
