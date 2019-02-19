use crate::core::traits::*;
use crate::core::{rescale_pixel_value, Image};
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

fn rescale_pixel<T>(x: f64) -> T
where
    T: FromPrimitive + Num + NumCast + PixelBound + Display,
{
    let tmax = T::max_pixel().to_f64().unwrap_or_else(|| 0.0f64);
    let tmin = T::min_pixel().to_f64().unwrap_or_else(|| 0.0f64);

    let x = x * (tmax - tmin) + tmin;
    T::from_f64(x).unwrap_or_else(|| T::zero())
}

pub fn rgb_to_hsv<T>(r: T, g: T, b: T) -> (T, T, T)
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
    let r_norm = rescale_pixel_value(r);
    let g_norm = rescale_pixel_value(g);
    let b_norm = rescale_pixel_value(b);
    let cmax = r_norm.max(g_norm.max(b_norm));
    let cmin = r_norm.min(g_norm.min(b_norm));
    let delta = cmax - cmin;

    let s = if cmax > 0.0f64 { delta / cmax } else { 0.0f64 };

    let h = if delta < std::f64::EPSILON {
        0.0 // hue is undefined for full black full white
    } else if cmax <= r_norm {
        60.0 * (((g_norm - b_norm) / delta) % 6.0)
    } else if cmax <= g_norm {
        60.0 * ((b_norm - r_norm) / delta + 2.0)
    } else {
        60.0 * ((r_norm - g_norm) / delta + 4.0)
    };
    let h = h / 360.0f64;

    let h = rescale_pixel(h);
    let s = rescale_pixel(s);
    let v = rescale_pixel(cmax);

    (h, s, v)
}

pub fn hsv_to_rgb<T>(h: T, s: T, v: T) -> (T, T, T)
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
    let h_deg = rescale_pixel_value(h) * 360.0f64;
    let s_norm = rescale_pixel_value(s);
    let v_norm = rescale_pixel_value(v);

    let c = v_norm * s_norm;
    let x = c * (1.0f64 - ((h_deg / 60.0f64) % 2.0f64 - 1.0f64).abs());
    let m = v_norm - c;

    let rgb = if 0.0f64 <= h_deg && h_deg < 60.0f64 {
        (c, x, 0.0f64)
    } else if 60.0f64 <= h_deg && h_deg < 120.0f64 {
        (x, c, 0.0f64)
    } else if 120.0f64 <= h_deg && h_deg < 180.0f64 {
        (0.0f64, c, x)
    } else if 180.0f64 <= h_deg && h_deg < 240.0f64 {
        (0.0f64, x, c)
    } else if 240.0f64 <= h_deg && h_deg < 300.0f64 {
        (x, 0.0f64, c)
    } else if 300.0f64 <= h_deg && h_deg < 360.0f64 {
        (c, 0.0f64, x)
    } else {
        (0.0f64, 0.0f64, 0.0f64)
    };

    let r = rescale_pixel(rgb.0 + m);
    let g = rescale_pixel(rgb.1 + m);
    let b = rescale_pixel(rgb.2 + m);

    (r, g, b)
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
            let r = pix[[0, 0, 0]];
            let g = pix[[0, 0, 1]];
            let b = pix[[0, 0, 2]];

            let (h, s, v) = rgb_to_hsv(r, g, b);
            res.slice_mut(s![i, j, ..]).assign(&arr1(&[h, s, v]));
        });
        Self::from_data(res)
    }
}

impl<T> From<Image<T, HSV>> for Image<T, RGB>
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
    fn from(image: Image<T, HSV>) -> Self {
        let mut res = Array3::<T>::zeros((image.rows(), image.cols(), RGB::channels()));
        let window = image.data.windows((1, 1, image.channels()));

        Zip::indexed(window).apply(|(i, j, _), pix| {
            let h = pix[[0, 0, 0]];
            let s = pix[[0, 0, 1]];
            let v = pix[[0, 0, 2]];

            let (r, g, b) = hsv_to_rgb(h, s, v);
            res.slice_mut(s![i, j, ..]).assign(&arr1(&[r, g, b]));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_rgb_hsv_check() {
        let mut i = Image::<u8, RGB>::new(1, 2);
        i.pixel_mut(0, 0).assign(&arr1(&[0, 0, 0]));
        i.pixel_mut(0, 1).assign(&arr1(&[255, 255, 255]));

        let hsv = Image::<u8, HSV>::from(i.clone());

        assert_eq!(hsv.pixel(0, 0)[[2]], 0);
        assert_eq!(hsv.pixel(0, 1)[[2]], 255);

        let rgb = Image::<u8, RGB>::from(hsv);
        assert_eq!(i, rgb);
    }

}
