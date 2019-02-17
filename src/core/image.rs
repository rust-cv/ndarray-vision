use crate::core::colour_models::*;
use ndarray::{s, Array3, ArrayView, ArrayView3, ArrayViewMut, Axis, Ix1, Zip};
use num_traits::{Num, NumAssignOps};
use num_traits::cast::{NumCast, FromPrimitive};
use std::fmt::Display;
use std::marker::PhantomData;

/// Basic structure containing an image.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Image<T, C=RGB> where C: ColourModel {
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
    T: Copy + Clone + FromPrimitive + Num + NumAssignOps + NumCast + PartialOrd + Display,
    C: ColourModel
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
            model: PhantomData
        }
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

impl <T, C>Image<T, C> where C: ColourModel {
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

    #[test] 
    fn image_consistency_checks() {
        let i = Image::<u8, RGB>::new(1, 2);
        assert_eq!(i.rows(), 1);
        assert_eq!(i.cols(), 2);
        assert_eq!(i.channels(), 3);
        assert_eq!(i.channels(), i.data.shape()[2]);
    }

}
