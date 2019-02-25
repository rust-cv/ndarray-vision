use crate::core::traits::*;
use crate::core::{normalise_pixel_value, Image};
use ndarray::{prelude::*, s, Zip};
use num_traits::cast::{FromPrimitive, NumCast};
use num_traits::{Num, NumAssignOps};
use std::convert::From;
use std::fmt::Display;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct Gray;
/// RGB colour as intended by sRGB and standardised in IEC 61966-2-1:1999
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
/// CIE XYZ standard - assuming a D50 reference white
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

impl RGB {
    /// Remove the gamma from a normalised channel
    pub fn remove_gamma(v: f64) -> f64 {
        if v < 0.04045 {
            v/12.92
        } else {
            ((v+0.055)/1.055).powf(2.4)
        }
    }

    /// Apply the gamma to a normalised channel
    pub fn apply_gamma(v: f64) -> f64 {
        if v < 0.0031308 {
            v * 12.92
        } else {
            1.055*v.powf(1.0/2.4) - 0.055
        }
    }
}


fn rescale_pixel<T>(x: f64) -> T
where
    T: FromPrimitive + Num + NumCast + PixelBound + Display,
{
    let tmax = T::max_pixel().to_f64().unwrap_or_else(|| 0.0f64);
    let tmin = T::min_pixel().to_f64().unwrap_or_else(|| 0.0f64);

    let x = x * (tmax - tmin) + tmin;

    T::from_f64(x).unwrap_or_else(T::zero)
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
    let r_norm = normalise_pixel_value(r);
    let g_norm = normalise_pixel_value(g);
    let b_norm = normalise_pixel_value(b);
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
    let h_deg = normalise_pixel_value(h) * 360.0f64;
    let s_norm = normalise_pixel_value(s);
    let v_norm = normalise_pixel_value(v);

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

impl<T> From<Image<T, RGB>> for Image<T, Gray>
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
        let mut res = Array3::<T>::zeros((image.rows(), image.cols(), Gray::channels()));
        let window = image.data.windows((1, 1, image.channels()));

        Zip::indexed(window).apply(|(i, j, _), pix| {
            let r = normalise_pixel_value(pix[[0, 0, 0]]);
            let g = normalise_pixel_value(pix[[0, 0, 1]]);
            let b = normalise_pixel_value(pix[[0, 0, 2]]);

            let gray = (0.3 * r) + (0.59 * g) + (0.11 * b);
            let gray = rescale_pixel(gray);

            res.slice_mut(s![i, j, ..]).assign(&arr1(&[gray]));
        });
        Self::from_data(res)
    }
}

impl<T> From<Image<T, Gray>> for Image<T, RGB>
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
    fn from(image: Image<T, Gray>) -> Self {
        let mut res = Array3::<T>::zeros((image.rows(), image.cols(), RGB::channels()));
        let window = image.data.windows((1, 1, image.channels()));

        Zip::indexed(window).apply(|(i, j, _), pix| {
            let gray = pix[[0, 0, 0]];

            res.slice_mut(s![i, j, ..])
                .assign(&arr1(&[gray, gray, gray]));
        });
        Self::from_data(res)
    }
}

impl<T> From<Image<T, RGB>> for Image<T, CIEXYZ>
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
        let mut res = Array3::<T>::zeros((image.rows(), image.cols(), CIEXYZ::channels()));
        let window = image.data.windows((1, 1, image.channels()));

        let m = arr2(&[[0.4360747, 0.3850649, 0.1430804],
                     [0.2225045, 0.7168786, 0.0606169],
                     [0.0139322, 0.0971045, 0.7141733]]);
        
        Zip::indexed(window).apply(|(i, j, _), pix| {
            let pixel = pix.index_axis(Axis(0), 0)
                           .index_axis(Axis(0), 0)
                           .mapv(normalise_pixel_value)
                           .mapv(RGB::remove_gamma);
            
            let pixel = m.dot(&pixel);
            let pixel = pixel.mapv(rescale_pixel);

            res.slice_mut(s![i, j, ..])
                .assign(&pixel);
        });
        Self::from_data(res)
    }
}

impl<T> From<Image<T, CIEXYZ>> for Image<T, RGB>
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
    fn from(image: Image<T, CIEXYZ>) -> Self {
        let mut res = Array3::<T>::zeros((image.rows(), image.cols(), RGB::channels()));
        let window = image.data.windows((1, 1, image.channels()));

        let m = arr2(&[[3.1338561, -1.6168667, -0.4906146],
                     [-0.9787684, 1.9161415, 0.0334540],
                     [0.0719453, -0.2289914, 1.4052427]]);

        Zip::indexed(window).apply(|(i, j, _), pix| {
            let pixel = pix.index_axis(Axis(0), 0)
                           .index_axis(Axis(0), 0)
                           .mapv(normalise_pixel_value);

            let pixel = m.dot(&pixel);

            let pixel = pixel.mapv(RGB::apply_gamma).mapv(rescale_pixel);

            res.slice_mut(s![i, j, ..])
                .assign(&pixel);
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
    use ndarray::s;
    use ndarray_stats::QuantileExt;
    use ndarray_rand::{RandomExt, F32};
    use rand::distributions::Uniform;

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

    #[test]
    fn gray_to_rgb_test() {
        let mut image = Image::<u8, Gray>::new(480, 640);
        let new_data = Array3::<u8>::random(image.data.dim(), Uniform::new(0, 255));
        image.data = new_data;

        let rgb = Image::<u8, RGB>::from(image.clone());
        let slice_2d = image.data.slice(s![.., .., 0]);

        assert_eq!(slice_2d, rgb.data.slice(s![.., .., 0]));
        assert_eq!(slice_2d, rgb.data.slice(s![.., .., 1]));
        assert_eq!(slice_2d, rgb.data.slice(s![.., .., 2]));
    }

    #[test]
    fn rgb_to_gray_basic() {
        // Check white, black, red, green, blue
        let data = vec![255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255];
        let image = Image::<u8, RGB>::from_shape_data(1, 5, data);

        let gray = Image::<u8, Gray>::from(image);

        // take standard 0.3 0.59 0.11 values and assume truncation
        let expected = vec![255, 0, 77, 150, 28];

        for (act, exp) in gray.data.iter().zip(expected.iter()) {
            let delta = (*act as i16 - *exp as i16).abs();
            assert!(delta < 2);
        }
    }

    #[test]
    fn basic_xyz_rgb_checks() {
        let mut image = Image::<f32, RGB>::new(100, 100);
        let new_data = Array3::<f32>::random(image.data.dim(), F32(Uniform::new(0.0, 1.0)));
        image.data = new_data;

        let xyz = Image::<f32, CIEXYZ>::from(image.clone());

        let rgb_restored = Image::<f32, RGB>::from(xyz);

        let mut delta = image.data - rgb_restored.data;
        delta.mapv_inplace(|x| x.abs());

        // 0.5% error in RGB -> XYZ -> RGB
        assert!(*delta.max().unwrap()*100.0 < 0.5);

    }

}
