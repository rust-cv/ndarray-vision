use crate::core::traits::*;
use crate::core::Image;
use ndarray::{arr1, s, Array3, Zip};
use num_traits::cast::{FromPrimitive, NumCast};
use num_traits::{Num, NumAssignOps};
use std::convert::From;
use std::fmt::Display;

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

/// Returns a normalised pixel value or 0 if it can't convert the types.
/// This should never fail if your types are good.
fn norm_pixel_value<T>(t: T) -> f64
where
    T: PixelBound + Num + NumCast,
{
    let numerator = (t + T::min_pixel()).to_f64();
    let denominator = (T::max_pixel() - T::min_pixel()).to_f64();

    let numerator = numerator.unwrap_or_else(|| 0.0f64);
    let denominator = denominator.unwrap_or_else(|| 1.0f64);

    numerator / denominator
}

impl<T> From<Image<T, RGB>> for Image<T, HSV>
where
    T: Copy
        + Clone
        + FromPrimitive
        + Num
        + NumAssignOps
        + NumCast
        + PartialOrd
        + Display
        + PixelBound,
{
    fn from(image: Image<T, RGB>) -> Self {
        let mut res = Array3::<T>::zeros((image.rows(), image.cols(), HSV::channels()));
        let window = image.data.windows((1, 1, image.channels()));

        Zip::indexed(window).apply(|(i, j, _), pix| {
            let r_norm = norm_pixel_value(pix[[0, 0, 0]]);
            let g_norm = norm_pixel_value(pix[[0, 0, 1]]);
            let b_norm = norm_pixel_value(pix[[0, 0, 2]]);
            let cmax = r_norm.max(g_norm.max(b_norm));
            let cmin = r_norm.min(g_norm.min(b_norm));
            let delta = cmax - cmin;

            let s = if cmax > 0.0f64 { delta / cmax } else { 0.0f64 };

            let h = if cmax <= r_norm {
                60.0 * (((g_norm - b_norm) / delta) % 6.0)
            } else if cmax <= g_norm {
                60.0 * ((b_norm - r_norm) / delta + 2.0)
            } else {
                60.0 * ((r_norm - g_norm) / delta + 4.0)
            };
            let h = h / 360.0f64;

            let h = h * T::max_pixel().to_f64().unwrap_or_else(|| 0.0f64);
            let h = T::from_f64(h).unwrap_or_else(|| T::zero());
            let s = s * T::max_pixel().to_f64().unwrap_or_else(|| 0.0f64);
            let s = T::from_f64(s).unwrap_or_else(|| T::zero());
            let v = cmax * T::max_pixel().to_f64().unwrap_or_else(|| 0.0f64);
            let v = T::from_f64(v).unwrap_or_else(|| T::zero());

            res.slice_mut(s![i, j, ..]).assign(&arr1(&[h, s, v]));
        });
        Self::from_data(res)
    }
}

impl ColourModel for RGB {}
impl ColourModel for HSV {}
impl ColourModel for HSI {}
impl ColourModel for HSL {}
impl ColourModel for YCrCb {}
impl ColourModel for CIEXYZ {}
impl ColourModel for CIELAB {}
impl ColourModel for CIELUV {}

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
