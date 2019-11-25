use crate::core::{ColourModel, Image};
use crate::transform::affine::translation;
use ndarray::{array, prelude::*, s, Data};
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
    /// Output type for the operation
    type Output;

    /// Transforms an image given the transformation matrix and output size.
    /// Assume nearest-neighbour interpolation
    fn transform(
        &self,
        transform: ArrayView2<f64>,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self::Output, Error>;
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Rect {
    x: isize,
    y: isize,
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

    let tl = (tl.0.round() as isize, tl.1.round() as isize);
    let tr = (tr.0.round() as isize, tr.1.round() as isize);
    let br = (br.0.round() as isize, br.1.round() as isize);
    let bl = (bl.0.round() as isize, bl.1.round() as isize);

    let leftmost = min(min(tl.0, tr.0), min(br.0, bl.0));
    let topmost = min(min(tl.1, tr.1), min(br.1, bl.1));
    let rightmost = max(max(tl.0, tr.0), max(br.0, bl.0));
    let bottommost = max(max(tl.1, tr.1), max(br.1, bl.1));
    Rect {
        x: leftmost,
        y: topmost,
        w: (rightmost - leftmost) as usize,
        h: (bottommost - topmost) as usize,
    }
}

impl<T, U> TransformExt for ArrayBase<U, Ix3>
where
    T: Copy + Clone + Num + NumAssignOps,
    U: Data<Elem = T>,
{
    type Output = Array3<T>;

    fn transform(
        &self,
        transform: ArrayView2<f64>,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self::Output, Error> {
        let shape = transform.shape();
        if !(shape[0] == 3 || shape[0] == 2) {
            Err(Error::InvalidTransformation)
        } else {
            let (mut result, new_transform) = match output_size {
                Some((r, c)) => (
                    Self::Output::zeros((r, c, self.shape()[2])),
                    transform.into_owned(),
                ),
                None => {
                    let dims = (self.shape()[0] as f64, self.shape()[1] as f64);
                    let bounds = bounding_box(dims, transform.view());
                    let new_trans = translation(bounds.x as f64, -bounds.y as f64).dot(&transform);
                    (
                        Self::Output::zeros((bounds.h, bounds.w, self.shape()[2])),
                        new_trans,
                    )
                }
            };

            let transform = new_transform
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
    type Output = Self;

    fn transform(
        &self,
        transform: ArrayView2<f64>,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self::Output, Error> {
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
    use std::f64::consts::PI;

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

    #[test]
    fn rotate() {
        let src = Image::<u8, Gray>::from_shape_data(5, 5, (0..25).collect());
        let trans = affine::rotate_around_centre(PI, (2.0, 2.0));
        let upside_down = src.transform(trans.view(), Some((5, 5))).unwrap();

        let res = upside_down.transform(trans.view(), Some((5, 5))).unwrap();

        assert_eq!(src, res);

        let trans_2 = affine::rotate_around_centre(PI / 2.0, (2.0, 2.0));
        let trans_3 = affine::rotate_around_centre(-PI / 2.0, (2.0, 2.0));

        let upside_down_sideways = upside_down.transform(trans_2.view(), Some((5, 5))).unwrap();
        let src_sideways = src.transform(trans_3.view(), Some((5, 5))).unwrap();

        assert_eq!(upside_down_sideways, src_sideways);
    }

    #[test]
    fn scale() {
        let src = Image::<u8, Gray>::from_shape_data(4, 4, (0..16).collect());
        let trans = affine::scale(0.5, 2.0);
        let res = src.transform(trans.view(), None).unwrap();

        assert_eq!(res.rows(), 8);
        assert_eq!(res.cols(), 2);
    }
}
