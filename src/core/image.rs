use crate::core::colour_models::*;
use ndarray::{s, Array3, ArrayView, ArrayView3, ArrayViewMut, Axis, Ix1, Zip};
use num_traits::{Num, NumAssignOps};
use num_traits::cast::{NumCast, FromPrimitive};
use std::fmt::Display;

/// Basic structure containing an image.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Image<T> {
    /// Images are always going to be 3D to handle rows, columns and colour
    /// channels
    ///
    /// This should allow for max compatibility with maths ops in ndarray
    pub data: Array3<T>,
    /// Representation of how colour is encoded in the image
    model: ColourModel,
}

impl<T> Image<T>
where
    T: Copy + Clone + FromPrimitive + Num + NumAssignOps + NumCast + PartialOrd + Display,
{
    //! Construct a new image filled with zeros using the given dimensions and
    //! a colour model
    pub fn new(rows: usize, columns: usize, model: ColourModel) -> Self {
        Image {
            data: Array3::<T>::zeros((rows, columns, model.channels())),
            model: model,
        }
    }

    pub fn from_shape_data(rows: usize, cols: usize, model: ColourModel, data: Vec<T>) -> Self {
        let data = Array3::<T>::from_shape_vec((rows, cols, model.channels()), data)
            .unwrap_or_else(|_| Array3::<T>::zeros((rows, cols, model.channels())));
        
        Image {
            data: data,
            model: model
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
    pub fn conv(&self, kernel: ArrayView3<T>) -> Image<T> {
        Image {
            data: conv(self.data.view(), kernel),
            model: self.model,
        }
    }

    /// Apply a convolution to the image
    pub fn conv_inplace(&mut self, kernel: ArrayView3<T>) {
        self.data = conv(self.data.view(), kernel);
    }

    pub fn convert_model(&self, model: ColourModel) -> Image<T> {
        unimplemented!()
    }
}

impl <T>Image<T> {
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
        self.model.channels()
    }

    /// Returns the colour model used by the image
    pub fn colour_model(&self) -> ColourModel {
        self.model
    }
    
    /// This method changes the colour model without changing any of the 
    /// underlying data
    pub fn force_model(&mut self, model: ColourModel) {
        self.model = model;
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
        let i = Image::<u8>::new(1, 2, ColourModel::RGB);
        assert_eq!(i.rows(), 1);
        assert_eq!(i.cols(), 2);
        assert_eq!(i.channels(), 3);
        assert_eq!(i.channels(), i.colour_model().channels());
    }

}
