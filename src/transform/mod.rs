use crate::core::{ColourModel, Image};
use ndarray::{array, prelude::*};
use num_traits::{Num, NumAssignOps};
use std::marker::PhantomData;

pub mod affine;

pub trait TransformExt {
    fn transform(&self, transform: ArrayView2<f64>, output_size: Option<(usize, usize)>) -> Self;
}

impl<T> TransformExt for Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps,
{
    fn transform(&self, transform: ArrayView2<f64>, output_size: Option<(usize, usize)>) -> Self {
        unimplemented!()
    }
}

impl<T> TransformExt for Image<T, C>
where
    T: Copy + Clone + Num + NumAssignOps,
    C: ColourModel,
{
    fn transform(&self, transform: ArrayView2<f64>, output_size: Option<(usize, usize)>) -> Self {
        let data = self.data.transform(transform, output_size);
        Self {
            data,
            model: PhantomData,
        }
    }
}
