use crate::core::{ColourModel, Image};
use ndarray::prelude::*;
use ndarray::{s, Zip};
use num_traits::{Num, NumAssignOps};
use std::marker::PhantomData;
use std::marker::Sized;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum Error {
    ChannelDimensionMismatch,
}

pub trait ConvolutionExt
where
    Self: Sized,
{
    type Data;
    fn conv2d(&self, kernel: ArrayView3<Self::Data>) -> Result<Self, Error>;

    fn conv2d_inplace(&mut self, kernel: ArrayView3<Self::Data>) -> Result<(), Error>;
}

impl<T> ConvolutionExt for Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps,
{
    type Data = T;

    fn conv2d(&self, kernel: ArrayView3<Self::Data>) -> Result<Self, Error> {
        if self.shape()[2] != kernel.shape()[2] {
            Err(Error::ChannelDimensionMismatch)
        } else {
            let mut result = Array3::<Self::Data>::zeros(self.dim());
            let k_s = kernel.shape();
            let row_offset = k_s[0] / 2;
            let col_offset = k_s[1] / 2;

            Zip::indexed(self.windows(kernel.dim())).apply(|(i, j, _), window| {
                let mult = &window * &kernel;
                let sums = mult.sum_axis(Axis(0)).sum_axis(Axis(0));
                result
                    .slice_mut(s![i + row_offset, j + col_offset, ..])
                    .assign(&sums);
            });
            Ok(result)
        }
    }

    fn conv2d_inplace(&mut self, kernel: ArrayView3<Self::Data>) -> Result<(), Error> {
        let data = self.conv2d(kernel)?;
        for (d, v) in self.indexed_iter_mut() {
            if let Some(d) = data.get(d) {
                *v = *d;
            }
        }
        Ok(())
    }
}

impl<T, C> ConvolutionExt for Image<T, C>
where
    T: Copy + Clone + Num + NumAssignOps,
    C: ColourModel,
{
    type Data = T;
    fn conv2d(&self, kernel: ArrayView3<Self::Data>) -> Result<Self, Error> {
        let data = self.data.conv2d(kernel)?;
        Ok(Self {
            data,
            model: PhantomData,
        })
    }

    fn conv2d_inplace(&mut self, kernel: ArrayView3<Self::Data>) -> Result<(), Error> {
        self.data.conv2d_inplace(kernel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::RGB;

    #[test]
    fn bad_dimensions() {
        let error = Err(Error::ChannelDimensionMismatch);
        let error2 = Err(Error::ChannelDimensionMismatch);

        let mut i = Image::<f64, RGB>::new(5, 5);
        let bad_kern = Array3::<f64>::zeros((2, 2, 2));
        assert_eq!(i.conv2d(bad_kern.view()), error);

        let data_clone = i.data.clone();
        let res = i.conv2d_inplace(bad_kern.view());
        assert_eq!(res, error2);
        assert_eq!(i.data, data_clone);

        let good_kern = Array3::<f64>::zeros((2,2, RGB::channels()));
        assert!(i.conv2d(good_kern.view()).is_ok());
        assert!(i.conv2d_inplace(good_kern.view()).is_ok());
    }
}