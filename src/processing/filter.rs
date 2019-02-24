use crate::core::{ColourModel, Image};
use ndarray::prelude::*;
use ndarray::{IntoDimension, Zip};
use ndarray_stats::interpolate::Nearest;
use ndarray_stats::Quantile1dExt;
use num_traits::{FromPrimitive, Num};
use std::marker::PhantomData;

pub trait MedianFilterExt {
    fn median_filter<E>(&self, region: E) -> Self
    where
        E: IntoDimension<Dim = Ix2>;
}

impl<T> MedianFilterExt for Array3<T>
where
    T: Copy + Clone + FromPrimitive + Num + Ord,
{
    fn median_filter<E>(&self, region: E) -> Self
    where
        E: IntoDimension<Dim = Ix2>,
    {
        let shape = region.into_dimension();
        let r_offset = shape[0] / 2;
        let c_offset = shape[1] / 2;
        let region = (shape[0], shape[1], 1);
        let mut result = Array3::<T>::zeros(self.dim());
        Zip::indexed(self.windows(region)).apply(|(i, j, k), window| {
            let mut flat_window = Array::from_iter(window.iter()).mapv(|x| *x);
            if let Some(r) = result.get_mut((i + r_offset, j + c_offset, k)) {
                if let Some(v) = flat_window.quantile_mut::<Nearest>(0.5) {
                    *r = v;
                }
            }
        });
        result
    }
}

impl<T, C> MedianFilterExt for Image<T, C>
where
    T: Copy + Clone + FromPrimitive + Num + Ord,
    C: ColourModel,
{
    fn median_filter<E>(&self, region: E) -> Self
    where
        E: IntoDimension<Dim = Ix2>,
    {
        let data = self.data.median_filter(region);
        Image {
            data: data,
            model: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::RGB;

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
}
