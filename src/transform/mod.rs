use crate::core::{ColourModel, Image};
use ndarray::{array, prelude::*, s};
use ndarray_linalg::solve::Inverse;
use num_traits::{Num, NumAssignOps};
use std::cmp::{max, min};
use std::marker::PhantomData;

pub mod affine;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub enum Error {
    InvalidTransformation,
    NonInvertibleTransformation,
}

pub trait TransformExt
where
    Self: Sized,
{
    fn transform(
        &self,
        transform: ArrayView2<f64>,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self, Error>;
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

fn source_coordinate(p: (f64, f64), trans: ArrayView2<f64>) -> (f64, f64) {
    let p = match trans.shape()[0] {
        2 => array![[p.0], [p.1]],
        3 => array![[p.0], [p.1], [1.0]],
        _ => unreachable!(),
    };

    let result = trans.dot(&p);
    let x = result[[0, 0]];
    let y = result[[1, 0]];
    let w = match trans.shape()[0] {
        2 => 1.0,
        3 => result[[2, 0]],
        _ => unreachable!(),
    };
    if (w - 1.0).abs() > std::f64::EPSILON {
        (x / w, y / w)
    } else {
        (x, y)
    }
}

fn bounding_box(dims: (f64, f64), transform: ArrayView2<f64>) -> Rect {
    let tl = source_coordinate((0.0, 0.0), transform);
    let tr = source_coordinate((0.0, dims.1), transform);
    let br = source_coordinate(dims, transform);
    let bl = source_coordinate((dims.0, 0.0), transform);

    let tl = (tl.0 as isize, tl.1 as isize);
    let tr = (tr.0 as isize, tr.1 as isize);
    let br = (br.0 as isize, br.1 as isize);
    let bl = (bl.0 as isize, bl.1 as isize);

    let mut leftmost = min(min(tl.0, tr.0), min(br.0, bl.0));
    let mut topmost = min(min(tl.1, tr.1), min(br.1, bl.1));
    let rightmost = max(max(tl.0, tr.0), max(br.0, bl.0));
    let bottommost = max(max(tl.1, tr.1), max(br.1, bl.1));
    if leftmost < 0 {
        leftmost = 0;
    } 
    if topmost < 0 {
        topmost = 0;
    }
    Rect {
        x: leftmost as usize,
        y: topmost as usize,
        w: (rightmost - leftmost) as usize,
        h: (bottommost - topmost) as usize,
    }
}

impl<T> TransformExt for Array3<T>
where
    T: Copy + Clone + Num + NumAssignOps,
{
    fn transform(
        &self,
        transform: ArrayView2<f64>,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self, Error> {
        let shape = transform.shape();
        if !(shape[0] == 3 || shape[0] == 2) {
            Err(Error::InvalidTransformation)
        } else {
            let mut result = match output_size {
                Some((r, c)) => Self::zeros((r, c, self.shape()[2])),
                None => {
                    let dims = (self.shape()[0] as f64, self.shape()[1] as f64);
                    let bounds = bounding_box(dims, transform.view());
                    Self::zeros((bounds.h, bounds.w, self.shape()[2]))
                }
            };

            let transform = transform
                .inv()
                .map_err(|_| Error::NonInvertibleTransformation)?;
            for r in 0..result.shape()[0] {
                for c in 0..result.shape()[1] {
                    let (x, y) = source_coordinate((c as f64, r as f64), transform.view());
                    let x = x.round() as isize;
                    let y = y.round() as isize;
                    if x >= 0
                        && y >= 0
                        && (x as usize) < self.shape()[1]
                        && (y as usize) < self.shape()[0]
                    {
                        result
                            .slice_mut(s![r, c, ..])
                            .assign(&self.slice(s![y, x, ..]));
                    } 
                }
            }
            Ok(result)
        }
    }
}

impl<T, C> TransformExt for Image<T, C>
where
    T: Copy + Clone + Num + NumAssignOps,
    C: ColourModel,
{
    fn transform(
        &self,
        transform: ArrayView2<f64>,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self, Error> {
        let data = self.data.transform(transform, output_size)?;
        Ok(Self {
            data,
            model: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate openblas_src;

    use super::affine;
    use super::*;
    use crate::core::colour_models::Gray;

    #[test]
    fn translation() {
        let src_data = vec![2.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 2.0, 3.0];
        let src = Image::<f64, Gray>::from_shape_data(3, 3, src_data);

        let trans = affine::translation(2.0, 1.0);

        let res = src.transform(trans.view(), Some((3, 3)));
        assert!(res.is_ok());
        let res = res.unwrap();

        let expected = vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let expected = Image::<f64, Gray>::from_shape_data(3, 3, expected);

        assert_eq!(expected, res)
    }
}
