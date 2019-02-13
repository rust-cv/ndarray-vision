use crate::core::formats::PixelFormat;
use ndarray::{ArrayView, ArrayViewMut, Array3, Ix1, s};
use num_traits::{Zero, One};

pub struct Image<T> {
    /// Images are always going to be 3D to handle rows, columns and colour 
    /// channels
    /// 
    /// This should allow for max compatibility with maths ops in ndarray
    pub data: Array3<T>,
    /// Pixel format stored internally
    format: PixelFormat,
}

impl<T> Image<T> where T: One, T: Zero, T: Clone {
    pub fn new(rows: usize, columns: usize, format: PixelFormat) -> Self {
        Image {
            data: Array3::<T>::zeros((rows, columns, format.channels())),
            format: format,
        }
    }

    pub fn pixel(&self, row: usize, col: usize) -> ArrayView<T, Ix1> {
        self.data.slice(s![row, col, ..])
    }

    pub fn pixel_mut(&mut self, row: usize, col: usize) -> ArrayViewMut<T, Ix1> {
        self.data.slice_mut(s![row, col, ..])
    }
}
