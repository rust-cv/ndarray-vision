use crate::core::{ColourModel, Image, ImageBase};
use ndarray::{prelude::*, s, Data};
use num_traits::{Num, NumAssignOps};
use std::fmt::Display;

pub mod affine;

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub enum TransformError {
    InvalidTransform,
    NonInvertibleTransform,
}

impl std::error::Error for TransformError {}

impl Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::InvalidTransform => write!(f, "invalid transform"),
            TransformError::NonInvertibleTransform => {
                write!(
                    f,
                    "Non Invertible Transform, Forward transform not yet implemented "
                )
            }
        }
    }
}

pub trait Transform {
    fn apply(&self, p: (f64, f64)) -> (f64, f64);
    fn apply_inverse(&self, p: (f64, f64)) -> (f64, f64);
    fn inverse_exists(&self) -> bool;
}

/// Composition of two transforms.  Specifically, derives transform2(transform1(image)).
/// this is not equivalent to running the transforms separately, since the composition of the
/// transforms occurs before sampling.  IE, running transforms separately incur a resample per
/// transform, whereas composed Transforms only incur a single image resample.
pub struct ComposedTransform<T: Transform> {
    transform1: T,
    transform2: T,
}

impl<T: Transform> Transform for ComposedTransform<T> {
    fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        self.transform2.apply(self.transform1.apply(p))
    }

    fn apply_inverse(&self, p: (f64, f64)) -> (f64, f64) {
        self.transform1
            .apply_inverse(self.transform2.apply_inverse(p))
    }

    fn inverse_exists(&self) -> bool {
        self.transform1.inverse_exists() && self.transform2.inverse_exists()
    }
}

pub trait TransformExt<T: Transform>
where
    Self: Sized,
{
    /// Output type for the operation
    type Output;

    /// Transforms an image given the transformation matrix and output size.
    /// Uses the source index coordinate space
    /// Assume nearest-neighbour interpolation
    fn transform(
        &self,
        transform: &T,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self::Output, TransformError>;
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Rect {
    x: isize,
    y: isize,
    w: usize,
    h: usize,
}

impl<T, U, V> TransformExt<V> for ArrayBase<U, Ix3>
where
    T: Copy + Clone + Num + NumAssignOps,
    U: Data<Elem = T>,
    V: Transform,
{
    type Output = Array<T, Ix3>;

    fn transform(
        &self,
        transform: &V,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self::Output, TransformError> {
        let mut output = match output_size {
            Some((r, c)) => Self::Output::zeros((r, c, self.shape()[2])),
            None => Self::Output::zeros(self.raw_dim()),
        };

        for r in 0..output.shape()[0] {
            for c in 0..output.shape()[1] {
                let (x, y) = transform.apply_inverse((c as f64, r as f64));
                let x = x.round() as isize;
                let y = y.round() as isize;
                if x >= 0
                    && y >= 0
                    && (x as usize) < self.shape()[1]
                    && (y as usize) < self.shape()[0]
                {
                    output
                        .slice_mut(s![r, c, ..])
                        .assign(&self.slice(s![y, x, ..]));
                }
            }
        }

        Ok(output)
    }
}

impl<T, U, C, V> TransformExt<V> for ImageBase<U, C>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps,
    C: ColourModel,
    V: Transform,
{
    type Output = Image<T, C>;

    fn transform(
        &self,
        transform: &V,
        output_size: Option<(usize, usize)>,
    ) -> Result<Self::Output, TransformError> {
        let data = self.data.transform(transform, output_size)?;
        let result = Self::Output::from_data(data).to_owned();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::affine;
    use super::*;
    use crate::core::colour_models::Gray;
    use std::f64::consts::PI;

    #[test]
    fn translation() {
        let src_data = vec![2.0, 0.0, 1.0, 0.0, 5.0, 0.0, 1.0, 2.0, 3.0];
        let src = Image::<f64, Gray>::from_shape_data(3, 3, src_data);

        let trans = affine::transform_from_2dmatrix(affine::translation(2.0, 1.0));

        let res = src.transform(&trans, Some((3, 3)));
        assert!(res.is_ok());
        let res = res.unwrap();

        let expected = vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let expected = Image::<f64, Gray>::from_shape_data(3, 3, expected);

        assert_eq!(expected, res)
    }

    #[test]
    fn rotate() {
        let src = Image::<u8, Gray>::from_shape_data(5, 5, (0..25).collect());
        let trans = affine::transform_from_2dmatrix(affine::rotate_around_centre(PI, (2.0, 2.0)));
        let upside_down = src.transform(&trans, Some((5, 5))).unwrap();

        let res = upside_down.transform(&trans, Some((5, 5))).unwrap();

        assert_eq!(src, res);

        let trans_2 =
            affine::transform_from_2dmatrix(affine::rotate_around_centre(PI / 2.0, (2.0, 2.0)));
        let trans_3 =
            affine::transform_from_2dmatrix(affine::rotate_around_centre(-PI / 2.0, (2.0, 2.0)));

        let upside_down_sideways = upside_down.transform(&trans_2, Some((5, 5))).unwrap();
        let src_sideways = src.transform(&trans_3, Some((5, 5))).unwrap();

        assert_eq!(upside_down_sideways, src_sideways);
    }

    #[test]
    fn scale() {
        let src = Image::<u8, Gray>::from_shape_data(4, 4, (0..16).collect());
        let trans = affine::transform_from_2dmatrix(affine::scale(0.5, 2.0));
        let res = src.transform(&trans, None).unwrap();

        assert_eq!(res.rows(), 4);
        assert_eq!(res.cols(), 4);
    }
}
