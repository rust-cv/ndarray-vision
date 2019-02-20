use crate::core::colour_models::*;
use crate::core::traits::PixelBound;
use ndarray::prelude::*;
use ndarray::{s, Zip};
use num_traits::cast::{FromPrimitive, NumCast};
use num_traits::{Num, NumAssignOps};
use std::fmt::Display;
use std::marker::PhantomData;

/// Basic structure containing an image.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Image<T, C>
where
    C: ColourModel,
{
    /// Images are always going to be 3D to handle rows, columns and colour
    /// channels
    ///
    /// This should allow for max compatibility with maths ops in ndarray
    pub data: Array3<T>,
    /// Representation of how colour is encoded in the image
    model: PhantomData<C>,
}

impl<T, C> Image<T, C>
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
    C: ColourModel,
{
    /// Construct the image from a given Array3
    pub fn from_data(data: Array3<T>) -> Self {
        Image {
            data: data,
            model: PhantomData,
        }
    }

    /// Construct a new image filled with zeros using the given dimensions and
    /// a colour model
    pub fn new(rows: usize, columns: usize) -> Self {
        Image {
            data: Array3::<T>::zeros((rows, columns, C::channels())),
            model: PhantomData,
        }
    }

    pub fn from_shape_data(rows: usize, cols: usize, data: Vec<T>) -> Self {
        let data = Array3::<T>::from_shape_vec((rows, cols, C::channels()), data)
            .unwrap_or_else(|_| Array3::<T>::zeros((rows, cols, C::channels())));

        Image {
            data: data,
            model: PhantomData,
        }
    }

    /// Converts image into a different type - doesn't scale to new pixel bounds
    pub fn into_type<T2>(&self) -> Image<T2, C>
    where
        T2: Copy
            + Clone
            + FromPrimitive
            + Num
            + NumAssignOps
            + NumCast
            + PartialOrd
            + Display
            + PixelBound
            + From<T>,
    {
        let rescale = |x: &T| {
            let scaled = rescale_pixel_value(*x)
                * (T2::max_pixel() - T2::min_pixel())
                    .to_f64()
                    .unwrap_or_else(|| 0.0f64);
            T2::from_f64(scaled).unwrap_or_else(|| T2::zero()) + T2::min_pixel()
        };
        let data = self.data.map(rescale);
        Image::<T2, C>::from_data(data)
    }

    /// Get a view of all colour channels at a pixels location
    pub fn pixel(&self, row: usize, col: usize) -> ArrayView<T, Ix1> {
        self.data.slice(s![row, col, ..])
    }

    /// Get a mutable view of a pixels colour channels given a location
    pub fn pixel_mut(&mut self, row: usize, col: usize) -> ArrayViewMut<T, Ix1> {
        self.data.slice_mut(s![row, col, ..])
    }

    /// Return a image formed when you convolve the image with a kernel
    pub fn conv(&self, kernel: ArrayView3<T>) -> Image<T, C> {
        Image {
            data: conv(self.data.view(), kernel),
            model: self.model,
        }
    }

    /// Apply a convolution to the image
    pub fn conv_inplace(&mut self, kernel: ArrayView3<T>) {
        self.data = conv(self.data.view(), kernel);
    }
}

impl<T, C> Image<T, C>
where
    C: ColourModel,
{
    /// Returns the number of rows in an image
    pub fn rows(&self) -> usize {
        self.data.shape()[0]
    }
    /// Returns the number of channels in an image
    pub fn cols(&self) -> usize {
        self.data.shape()[1]
    }

    /// Convenience method to get number of channels
    pub fn channels(&self) -> usize {
        C::channels()
    }
}

/// Returns a normalised pixel value or 0 if it can't convert the types.
/// This should never fail if your types are good.
pub fn rescale_pixel_value<T>(t: T) -> f64
where
    T: PixelBound + Num + NumCast,
{
    let numerator = (t + T::min_pixel()).to_f64();
    let denominator = (T::max_pixel() - T::min_pixel()).to_f64();

    let numerator = numerator.unwrap_or_else(|| 0.0f64);
    let denominator = denominator.unwrap_or_else(|| 1.0f64);

    numerator / denominator
}

/// Implements a simple image convolution given a image and kernel
/// TODO Add an option to change kernel centre
pub fn conv<T>(image: ArrayView3<T>, kernel: ArrayView3<T>) -> Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps,
{
    let mut result = Array3::<T>::zeros(image.dim());
    let k_s = kernel.shape();
    let row_offset = k_s[0] / 2;
    let col_offset = k_s[1] / 2;

    Zip::indexed(image.windows(kernel.dim())).apply(|(i, j, _), window| {
        let mult = &window * &kernel;
        let sums = mult.sum_axis(Axis(0)).sum_axis(Axis(0));
        result
            .slice_mut(s![i + row_offset, j + col_offset, ..])
            .assign(&sums);
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn image_consistency_checks() {
        let i = Image::<u8, RGB>::new(1, 2);
        assert_eq!(i.rows(), 1);
        assert_eq!(i.cols(), 2);
        assert_eq!(i.channels(), 3);
        assert_eq!(i.channels(), i.data.shape()[2]);
    }

    #[test]
    fn image_type_conversion() {
        let mut i = Image::<u8, RGB>::new(1, 1);
        i.pixel_mut(0, 0)
            .assign(&arr1(&[u8::max_value(), 0, u8::max_value() / 3]));
        let t: Image<u16, RGB> = i.into_type();
        assert_eq!(
            t.pixel(0, 0),
            arr1(&[u16::max_value(), 0, u16::max_value() / 3])
        );
    }

}
