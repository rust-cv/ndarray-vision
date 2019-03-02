use crate::core::*;
use crate::processing::*;
use core::ops::Neg;
use ndarray::prelude::*;
use num_traits::{cast::FromPrimitive, real::Real, Num, NumAssignOps};
use std::marker::Sized;

pub trait SobelExt
where
    Self: Sized,
{
    type Output;

    fn apply_sobel(&self) -> Result<Self::Output, conv::Error>;
}

impl<T> SobelExt for Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
{
    type Output = Self;

    fn apply_sobel(&self) -> Result<Self::Output, conv::Error> {
        let v_temp: Array3<T> = SobelFilter::build_with_params(Orientation::Vertical).unwrap();
        let h_temp: Array3<T> = SobelFilter::build_with_params(Orientation::Horizontal).unwrap();
        let shape = (v_temp.shape()[0], v_temp.shape()[1], self.shape()[2]);
        let h_kernel = Array3::<T>::from_shape_fn(shape, |(i, j, _)| h_temp[[i, j, 0]]);
        let v_kernel = Array3::<T>::from_shape_fn(shape, |(i, j, _)| v_temp[[i, j, 0]]);

        let h_deriv = self.conv2d(h_kernel.view())?;
        let v_deriv = self.conv2d(v_kernel.view())?;

        let h_deriv = h_deriv.mapv(|x| x.powi(2));
        let v_deriv = v_deriv.mapv(|x| x.powi(2));

        let result = h_deriv + v_deriv;

        let result = result.mapv(|x| x.sqrt());
        Ok(result)
    }
}

impl<T, C> SobelExt for Image<T, C>
where
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
    C: ColourModel,
{
    type Output = Self;

    fn apply_sobel(&self) -> Result<Self::Output, conv::Error> {
        let data = self.data.apply_sobel()?;
        Ok(Image::from_data(data))
    }
}
