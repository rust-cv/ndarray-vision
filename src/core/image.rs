use crate::core::formats::PixelFormat;
use ndarray::{ArrayView, ArrayView3, ArrayViewMut, Array3, Axis, Ix1, s, Zip};
use num_traits::{Zero, One};


pub struct Image<T> {
    /// Images are always going to be 3D to handle rows, columns and colour 
    /// channels
    ///
    /// 
    /// This should allow for max compatibility with maths ops in ndarray
    pub data: Array3<T>,
    /// Pixel format stored internally
    format: PixelFormat,
}

impl<T> Image<T> where T: One, T: Zero, T: Clone, T: Default {
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

    pub fn conv(&self, kernel: ArrayView3<T>) -> Image<T> {
        Image {
            data: conv(self.data.view(), kernel),
            format: self.format
        }
    }

    pub fn conv_inplace(&mut self, kernel: ArrayView3<T>) {
        self.data = conv(self.data.view(), kernel);
    }
}


pub fn conv<T>(image: ArrayView3<T>, kernel: ArrayView3<T>) -> Array3<T> where T: Default, T: Clone {
    let mut result = Array3::<T>::default(image.dim());
    let k_s = kernel.shape();
    let row_offset = k_s[0]/2;
    let col_offset = k_s[1]/2;

    Zip::indexed(image.windows(kernel.dim()))
        .apply(|(i, j, _), window| {
            let mult = window * kernel;
            let sums = mult.sum_axis(Axis(2));
            result.slice_mut(s![i+row_offset, j+col_offset, ..]).assign(sums);
        });

    result
}
