use crate::core::*;
use crate::processing::*;
use core::mem::MaybeUninit;
use core::ops::Neg;
use ndarray::{prelude::*, s, DataMut, OwnedRepr, Zip};
use num_traits::{cast::FromPrimitive, real::Real, Num, NumAssignOps};
use std::marker::Sized;

/// Runs the sobel operator on an image
pub trait SobelExt
where
    Self: Sized,
{
    /// Type to output
    type Output;
    /// Returns the magnitude output of the sobel - an image of only lines
    fn apply_sobel(&self) -> Result<Self::Output, Error>;

    /// Returns the magntitude and rotation outputs for use in other algorithms
    /// like the Canny edge detector. Rotation is in radians
    fn full_sobel(&self) -> Result<(Self::Output, Self::Output), Error>;
}

fn get_edge_images<T, U>(mat: &ArrayBase<U, Ix3>) -> Result<(Array3<T>, Array3<T>), Error>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
{
    let v_temp: Array3<T> = SobelFilter::build_with_params(Orientation::Vertical).unwrap();
    let h_temp: Array3<T> = SobelFilter::build_with_params(Orientation::Horizontal).unwrap();
    let shape = (v_temp.shape()[0], v_temp.shape()[1], mat.shape()[2]);
    let mut h_kernel = Array3::<T>::uninit(shape);
    let mut v_kernel = Array3::<T>::uninit(shape);
    for i in 0..mat.dim().2 {
        h_temp
            .slice(s![.., .., 0])
            .assign_to(h_kernel.slice_mut(s![.., .., i]));
        v_temp
            .slice(s![.., .., 0])
            .assign_to(v_kernel.slice_mut(s![.., .., i]));
    }
    let h_kernel = unsafe { h_kernel.assume_init() };
    let v_kernel = unsafe { v_kernel.assume_init() };
    let h_deriv = mat.conv2d(h_kernel.view())?;
    let v_deriv = mat.conv2d(v_kernel.view())?;

    Ok((h_deriv, v_deriv))
}

impl<T, U> SobelExt for ArrayBase<U, Ix3>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
{
    type Output = ArrayBase<OwnedRepr<T>, Ix3>;

    fn apply_sobel(&self) -> Result<Self::Output, Error> {
        let (h_deriv, v_deriv) = get_edge_images(self)?;
        let res_shape = h_deriv.dim();
        let mut result = Self::Output::uninit(res_shape);
        for r in 0..res_shape.0 {
            for c in 0..res_shape.1 {
                for channel in 0..res_shape.2 {
                    let mut temp = (h_deriv[[r, c, channel]].powi(2)
                        + v_deriv[[r, c, channel]].powi(2))
                    .sqrt();
                    if temp > T::one() {
                        temp = T::one();
                    }
                    unsafe {
                        *result.uget_mut([r, c, channel]) = MaybeUninit::new(temp);
                    }
                }
            }
        }
        Ok(unsafe { result.assume_init() })
    }

    fn full_sobel(&self) -> Result<(Self::Output, Self::Output), Error> {
        let (h_deriv, v_deriv) = get_edge_images(self)?;

        let mut magnitude = h_deriv.mapv(|x| x.powi(2)) + v_deriv.mapv(|x| x.powi(2));
        magnitude.mapv_inplace(|x| x.sqrt());
        magnitude.mapv_inplace(|x| if x > T::one() { T::one() } else { x });

        let dim = h_deriv.dim();
        let mut rotation = Array3::uninit((dim.0, dim.1, dim.2));
        Zip::from(&mut rotation)
            .and(&h_deriv)
            .and(&v_deriv)
            .for_each(|r, &h, &v| *r = MaybeUninit::new(v.atan2(h)));

        let rotation = unsafe { rotation.assume_init() };

        Ok((magnitude, rotation))
    }
}

impl<T, U, C> SobelExt for ImageBase<U, C>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
    C: ColourModel,
{
    type Output = Image<T, C>;

    fn apply_sobel(&self) -> Result<Self::Output, Error> {
        let data = self.data.apply_sobel()?;
        Ok(Image::from_data(data))
    }

    fn full_sobel(&self) -> Result<(Self::Output, Self::Output), Error> {
        self.data
            .full_sobel()
            .map(|(m, r)| (Image::from_data(m), Image::from_data(r)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_ulps_eq;

    #[test]
    fn simple() {
        let mut image: Image<f64, Gray> = ImageBase::new(11, 11);
        image.data.slice_mut(s![4..6, 4..6, ..]).fill(1.0);
        image.data.slice_mut(s![3..7, 5, ..]).fill(1.0);
        image.data.slice_mut(s![5, 3..7, ..]).fill(1.0);

        let sobel = image.clone().full_sobel().unwrap();

        println!("Source: {:?}", image);
        println!("mag: {:?}", sobel.0);
        println!("rot: {:?}", sobel.1);
        // Did a calculation of sobel_mag[1..9, 1..9, ..]
        let mag = vec![
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.4142135623731,
            2.0,
            1.4142135623731,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.4142135623731,
            4.24264068711929,
            4.0,
            4.24264068711929,
            1.4142135623731,
            0.0,
            0.0,
            0.0,
            1.4142135623731,
            4.24264068711929,
            4.24264068711929,
            2.0,
            4.24264068711929,
            4.24264068711929,
            1.4142135623731,
            0.0,
            0.0,
            2.0,
            4.0,
            2.0,
            0.0,
            2.0,
            4.0,
            2.0,
            0.0,
            0.0,
            1.4142135623731,
            4.24264068711929,
            4.24264068711929,
            2.0,
            4.24264068711929,
            4.24264068711929,
            1.4142135623731,
            0.0,
            0.0,
            0.0,
            1.4142135623731,
            4.24264068711929,
            4.0,
            4.24264068711929,
            1.4142135623731,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.4142135623731,
            2.0,
            1.4142135623731,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        println!("len: {}", mag.len());

        let mag = Array::from_shape_vec((9, 9), mag).unwrap();

        assert_ulps_eq!(sobel.0.data.slice(s![1..10, 1..10, 0]), mag);
    }
}
