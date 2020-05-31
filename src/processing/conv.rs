use crate::core::padding::*;
use crate::core::{kernel_centre, ColourModel, Image, ImageBase};
use crate::processing::Error;
use ndarray::prelude::*;
use ndarray::{s, Data, DataMut, Zip};
use num_traits::{Num, NumAssignOps};
use std::marker::PhantomData;
use std::marker::Sized;

/// Perform image convolutions
pub trait ConvolutionExt<T: Copy>
where
    Self: Sized,
{
    /// Type for the output as data will have to be allocated
    type Output;

    /// Perform a convolution returning the resultant data
    /// applies the default padding of zero padding
    fn conv2d<U: Data<Elem = T>>(&self, kernel: ArrayBase<U, Ix3>) -> Result<Self::Output, Error>;
    /// Performs the convolution inplace mutating the containers data
    /// applies the default padding of zero padding
    fn conv2d_inplace<U: Data<Elem = T>>(&mut self, kernel: ArrayBase<U, Ix3>)
        -> Result<(), Error>;
    /// Perform a convolution returning the resultant data
    /// applies the default padding of zero padding
    fn conv2d_with_padding<U: Data<Elem = T>>(
        &self,
        kernel: ArrayBase<U, Ix3>,
        strategy: &impl PaddingStrategy<T>,
    ) -> Result<Self::Output, Error>;
    /// Performs the convolution inplace mutating the containers data
    /// applies the default padding of zero padding
    fn conv2d_inplace_with_padding<U: Data<Elem = T>>(
        &mut self,
        kernel: ArrayBase<U, Ix3>,
        strategy: &impl PaddingStrategy<T>,
    ) -> Result<(), Error>;
}

impl<T, U> ConvolutionExt<T> for ArrayBase<U, Ix3>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps,
{
    type Output = Array<T, Ix3>;

    fn conv2d<B: Data<Elem = T>>(&self, kernel: ArrayBase<B, Ix3>) -> Result<Self::Output, Error> {
        self.conv2d_with_padding(kernel, &ZeroPadding {})
    }

    fn conv2d_inplace<B: Data<Elem = T>>(
        &mut self,
        kernel: ArrayBase<B, Ix3>,
    ) -> Result<(), Error> {
        self.assign(&self.conv2d_with_padding(kernel, &ZeroPadding {})?);
        Ok(())
    }

    #[inline]
    fn conv2d_with_padding<B: Data<Elem = T>>(
        &self,
        kernel: ArrayBase<B, Ix3>,
        strategy: &impl PaddingStrategy<T>,
    ) -> Result<Self::Output, Error> {
        if self.shape()[2] != kernel.shape()[2] {
            Err(Error::ChannelDimensionMismatch)
        } else {
            let k_s = kernel.shape();
            // Bit icky but handles fact that uncentred convolutions will cross the bounds
            // otherwise
            let (row_offset, col_offset) = kernel_centre(k_s[0], k_s[1]);
            let shape = (self.shape()[0], self.shape()[1], self.shape()[2]);

            if shape.0 > 0 && shape.1 > 0 {
                let mut result = unsafe { Self::Output::uninitialized(shape) };
                let tmp = self.pad((row_offset, col_offset), strategy);

                Zip::indexed(tmp.windows(kernel.dim())).apply(|(i, j, _), window| {
                    let mut pixel = vec![T::zero(); shape.2];
                    let mut temp;
                    for channel in 0..k_s[2] {
                        temp = T::zero();
                        for r in 0..k_s[0] {
                            for c in 0..k_s[1] {
                                temp += window[[r, c, channel]] * kernel[[r, c, channel]];
                            }
                        }
                        pixel[channel] = temp;
                    }
                    result
                        .slice_mut(s![i, j, ..])
                        .assign(&ArrayView1::from(&pixel));
                });
                Ok(result)
            } else {
                Err(Error::InvalidDimensions)
            }
        }
    }

    fn conv2d_inplace_with_padding<B: Data<Elem = T>>(
        &mut self,
        kernel: ArrayBase<B, Ix3>,
        strategy: &impl PaddingStrategy<T>,
    ) -> Result<(), Error> {
        self.assign(&self.conv2d_with_padding(kernel, strategy)?);
        Ok(())
    }
}

impl<T, U, C> ConvolutionExt<T> for ImageBase<U, C>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps,
    C: ColourModel,
{
    type Output = Image<T, C>;

    fn conv2d<B: Data<Elem = T>>(&self, kernel: ArrayBase<B, Ix3>) -> Result<Self::Output, Error> {
        let data = self.data.conv2d(kernel)?;
        Ok(Self::Output {
            data,
            model: PhantomData,
        })
    }

    fn conv2d_inplace<B: Data<Elem = T>>(
        &mut self,
        kernel: ArrayBase<B, Ix3>,
    ) -> Result<(), Error> {
        self.data.conv2d_inplace(kernel)
    }

    fn conv2d_with_padding<B: Data<Elem = T>>(
        &self,
        kernel: ArrayBase<B, Ix3>,
        strategy: &impl PaddingStrategy<T>,
    ) -> Result<Self::Output, Error> {
        let data = self.data.conv2d_with_padding(kernel, strategy)?;
        Ok(Self::Output {
            data,
            model: PhantomData,
        })
    }

    fn conv2d_inplace_with_padding<B: Data<Elem = T>>(
        &mut self,
        kernel: ArrayBase<B, Ix3>,
        strategy: &impl PaddingStrategy<T>,
    ) -> Result<(), Error> {
        self.data.conv2d_inplace_with_padding(kernel, strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::{Gray, RGB};
    use ndarray::arr3;

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

        let good_kern = Array3::<f64>::zeros((2, 2, RGB::channels()));
        assert!(i.conv2d(good_kern.view()).is_ok());
        assert!(i.conv2d_inplace(good_kern.view()).is_ok());
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn basic_conv() {
        let input_pixels = vec![
            1, 1, 1, 0, 0,
            0, 1, 1, 1, 0,
            0, 0, 1, 1, 1,
            0, 0, 1, 1, 0,
            0, 1, 1, 0, 0,
        ];
        let output_pixels = vec![
            2, 2, 3, 1, 1,
            1, 4, 3, 4, 1,
            1, 2, 4, 3, 3,
            1, 2, 3, 4, 1,
            0, 2, 2, 1, 1, 
        ];

        let kern = arr3(
            &[
                [[1], [0], [1]],
                [[0], [1], [0]],
                [[1], [0], [1]]
            ]);

        let input = Image::<u8, Gray>::from_shape_data(5, 5, input_pixels);
        let expected = Image::<u8, Gray>::from_shape_data(5, 5, output_pixels);

        assert_eq!(Ok(expected), input.conv2d(kern.view()));
    }

    #[test]
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn basic_conv_inplace() {
        let input_pixels = vec![
            1, 1, 1, 0, 0,
            0, 1, 1, 1, 0,
            0, 0, 1, 1, 1,
            0, 0, 1, 1, 0,
            0, 1, 1, 0, 0,
        ];

        let output_pixels = vec![
            2, 2, 3, 1, 1,
            1, 4, 3, 4, 1,
            1, 2, 4, 3, 3,
            1, 2, 3, 4, 1,
            0, 2, 2, 1, 1,
        ];

        let kern = arr3(
            &[
                [[1], [0], [1]],
                [[0], [1], [0]],
                [[1], [0], [1]]
            ]);

        let mut input = Image::<u8, Gray>::from_shape_data(5, 5, input_pixels);
        let expected = Image::<u8, Gray>::from_shape_data(5, 5, output_pixels);

        input.conv2d_inplace(kern.view()).unwrap();

        assert_eq!(expected, input);
    }
}
