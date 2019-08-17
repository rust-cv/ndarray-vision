use crate::core::{ColourModel, Image};
use ndarray::{prelude::*, s};
use std::marker::PhantomData;

/// Defines a method for padding the data of an image applied directly to the
/// ndarray type internally. Padding is symmetric
pub trait PaddingStrategy<T>
where
    T: Copy,
{
    /// Taking in the image data and the margin to apply to rows and columns
    /// returns a padded image
    fn pad(&self, image: ArrayView3<T>, padding: (usize, usize)) -> Array3<T>;
}

/// Doesn't apply any padding to the image returning it unaltered regardless
/// of padding value
#[derive(Clone, Copy, Eq, PartialEq, Hash, Debug)]
pub struct NoPadding;

/// Pad the image with a constant value
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct ConstantPadding<T>(T)
where
    T: Copy;

impl<T> PaddingStrategy<T> for NoPadding
where
    T: Copy + Sized,
{
    fn pad(&self, image: ArrayView3<T>, _padding: (usize, usize)) -> Array3<T> {
        image.to_owned()
    }
}

impl<T> PaddingStrategy<T> for ConstantPadding<T>
where
    T: Copy + Sized,
{
    fn pad(&self, image: ArrayView3<T>, padding: (usize, usize)) -> Array3<T> {
        let shape = (
            image.shape()[0] + padding.0 * 2,
            image.shape()[1] + padding.1 * 2,
            image.shape()[2],
        );

        let mut result = Array::from_elem(shape, self.0);
        result
            .slice_mut(s![
                padding.0..shape.0 - padding.0,
                padding.1..shape.1 - padding.1,
                ..
            ])
            .assign(&image);

        result
    }
}

/// Padding extension for images
pub trait PaddingExt
where
    Self: Sized,
{
    /// Data type for container
    type Data;
    /// Pad the object with the given padding and strategy
    fn pad(&self, padding: (usize, usize), strategy: &dyn PaddingStrategy<Self::Data>) -> Self;
}

impl<T> PaddingExt for Array3<T>
where
    T: Copy + Sized,
{
    type Data = T;

    fn pad(&self, padding: (usize, usize), strategy: &dyn PaddingStrategy<Self::Data>) -> Self {
        strategy.pad(self.view(), padding)
    }
}

impl<T, C> PaddingExt for Image<T, C>
where
    T: Copy + Sized,
    C: ColourModel,
{
    type Data = T;

    fn pad(&self, padding: (usize, usize), strategy: &dyn PaddingStrategy<Self::Data>) -> Self {
        Self {
            data: strategy.pad(self.data.view(), padding),
            model: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::{Gray, RGB};

    #[test]
    fn constant_padding() {
        let i = Image::<u8, Gray>::from_shape_data(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let p = i.pad((1, 1), &ConstantPadding(0));

        let exp = Image::<u8, Gray>::from_shape_data(
            5,
            5,
            vec![
                0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
            ],
        );
        assert_eq!(p, exp);

        let p = i.pad((1, 1), &ConstantPadding(2));

        let exp = Image::<u8, Gray>::from_shape_data(
            5,
            5,
            vec![
                2, 2, 2, 2, 2, 2, 1, 2, 3, 2, 2, 4, 5, 6, 2, 2, 7, 8, 9, 2, 2, 2, 2, 2, 2,
            ],
        );
        assert_eq!(p, exp);

        let p = i.pad((2, 0), &ConstantPadding(0));

        let exp = Image::<u8, Gray>::from_shape_data(
            7,
            3,
            vec![
                0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,
            ],
        );
        assert_eq!(p, exp);
    }

    #[test]
    fn no_padding() {
        let i = Image::<u8, RGB>::new(5, 5);
        let p = i.pad((10, 10), &NoPadding {});

        assert_eq!(i, p);

        let p = i.pad((0, 0), &NoPadding {});
        assert_eq!(i, p);
    }
}
