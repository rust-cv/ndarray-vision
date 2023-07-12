use crate::core::{ColourModel, Image, ImageBase};
use ndarray::prelude::*;
use ndarray::{Data, IntoDimension, OwnedRepr, Zip};
use ndarray_stats::interpolate::*;
use ndarray_stats::Quantile1dExt;
use noisy_float::types::n64;
use num_traits::{FromPrimitive, Num, ToPrimitive};
use std::marker::PhantomData;

/// Median filter, given a region to move over the image, each pixel is given
/// the median value of itself and it's neighbours
pub trait MedianFilterExt {
    type Output;
    /// Run the median filter given the region. Median is assumed to be calculated
    /// independently for each channel.
    fn median_filter<E>(&self, region: E) -> Self::Output
    where
        E: IntoDimension<Dim = Ix2>;
}

impl<T, U> MedianFilterExt for ArrayBase<U, Ix3>
where
    U: Data<Elem = T>,
    T: Copy + Clone + FromPrimitive + ToPrimitive + Num + Ord,
{
    type Output = ArrayBase<OwnedRepr<T>, Ix3>;

    fn median_filter<E>(&self, region: E) -> Self::Output
    where
        E: IntoDimension<Dim = Ix2>,
    {
        let shape = region.into_dimension();
        let r_offset = shape[0] / 2;
        let c_offset = shape[1] / 2;
        let region = (shape[0], shape[1], 1);
        let mut result = Array3::<T>::zeros(self.dim());
        Zip::indexed(self.windows(region)).for_each(|(i, j, k), window| {
            let mut flat_window = Array::from_iter(window.iter()).mapv(|x| *x);
            if let Ok(v) = flat_window.quantile_mut(n64(0.5f64), &Linear {}) {
                if let Some(r) = result.get_mut([i + r_offset, j + c_offset, k]) {
                    *r = v;
                }
            }
        });
        result
    }
}

impl<T, U, C> MedianFilterExt for ImageBase<U, C>
where
    U: Data<Elem = T>,
    T: Copy + Clone + FromPrimitive + ToPrimitive + Num + Ord,
    C: ColourModel,
{
    type Output = Image<T, C>;

    fn median_filter<E>(&self, region: E) -> Self::Output
    where
        E: IntoDimension<Dim = Ix2>,
    {
        let data = self.data.median_filter(region);
        Image {
            data,
            model: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::{Gray, RGB};

    #[test]
    fn simple_median() {
        let mut pixels = Vec::<u8>::new();
        for i in 0..9 {
            pixels.extend_from_slice(&[i, i + 1, i + 2]);
        }
        let image = Image::<_, RGB>::from_shape_data(3, 3, pixels);

        let image = image.median_filter((3, 3));

        let mut expected = Image::<u8, RGB>::new(3, 3);
        expected.pixel_mut(1, 1).assign(&arr1(&[4, 5, 6]));

        assert_eq!(image, expected);
    }

    #[test]
    fn row_median() {
        let pixels = vec![1, 2, 3, 4, 5, 6, 7];
        let image = Image::<_, Gray>::from_shape_data(7, 1, pixels);
        let image = image.median_filter((3, 1));

        let expected_pixels = vec![0, 2, 3, 4, 5, 6, 0];
        let expected = Image::<_, Gray>::from_shape_data(7, 1, expected_pixels);

        assert_eq!(image, expected);
    }
}
