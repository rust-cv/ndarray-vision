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
    /// Returns the magnitude output of the sobel - an image of only lines
    fn apply_sobel(&self) -> Result<Self, Error>;
    
    /// Returns the magntitude and rotation outputs for use in other algorithms
    /// like the Canny edge detector
    fn full_sobel(&self) -> Result<(Self::Output, Self::Output), Error>;
}

fn get_edge_images<T>(mat: &Array3<T>) -> Result<(Array3<T>, Array3<T>), Error>
where
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
{
    let v_temp: Array3<T> = SobelFilter::build_with_params(Orientation::Vertical).unwrap();
    let h_temp: Array3<T> = SobelFilter::build_with_params(Orientation::Horizontal).unwrap();
    let shape = (v_temp.shape()[0], v_temp.shape()[1], mat.shape()[2]);
    let h_kernel = Array3::<T>::from_shape_fn(shape, |(i, j, _)| h_temp[[i, j, 0]]);
    let v_kernel = Array3::<T>::from_shape_fn(shape, |(i, j, _)| v_temp[[i, j, 0]]);

    let h_deriv = mat.conv2d(h_kernel.view())?;
    let v_deriv = mat.conv2d(v_kernel.view())?;
    
    Ok((h_deriv, v_deriv))
}

impl<T> SobelExt for Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
{
    type Output = Self;

    fn apply_sobel(&self) -> Result<Self, Error> {
        let (h_deriv, v_deriv) = get_edge_images(self)?;

        let h_deriv = h_deriv.mapv(|x| x.powi(2));
        let v_deriv = v_deriv.mapv(|x| x.powi(2));

        let mut result = h_deriv + v_deriv;
        result.mapv_inplace(|x| x.sqrt());
        Ok(result)
    }
    
    fn full_sobel(&self) -> Result<(Self::Output, Self::Output), Error> {
        let (h_deriv, v_deriv) = get_edge_images(self)?;

        let mut magnitude = h_deriv.mapv(|x| x.powi(2)) + v_deriv.mapv(|x| x.powi(2));
        magnitude.mapv_inplace(|x| x.sqrt());

        let mut rotation = v_deriv/h_deriv;
        rotation.mapv_inplace(|x| x.atan());
        
        Ok((magnitude, rotation))
    }
}

impl<T, C> SobelExt for Image<T, C>
where
    T: Copy + Clone + Num + NumAssignOps + Neg<Output = T> + FromPrimitive + Real,
    C: ColourModel,
{
    type Output = Array3<T>;

    fn apply_sobel(&self) -> Result<Self, Error> {
        let data = self.data.apply_sobel()?;
        Ok(Image::from_data(data))
    }
    
    fn full_sobel(&self) -> Result<(Self::Output, Self::Output), Error> {
        self.data.full_sobel()
    }
}
