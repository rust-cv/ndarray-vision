use core::ops::Neg;
use ndarray::prelude::*;
use ndarray::IntoDimension;
use num_traits::{cast::FromPrimitive, float::Float, Num, NumAssignOps};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Error {
    InvalidDimensions,
    InvalidParameter,
    NumericError,
}

pub trait KernelBuilder<T> {
    type Params;
    /// Build a kernel with a given dimension
    fn build<D>(shape: D) -> Result<Array3<T>, Error>
    where
        D: Copy + IntoDimension<Dim = Ix3>;
    /// For kernels with optional parameters use build with params otherwise
    /// appropriate default parameters will be chosen
    fn build_with_params<D>(shape: D, _p: Self::Params) -> Result<Array3<T>, Error>
    where
        D: Copy + IntoDimension<Dim = Ix3>,
    {
        Self::build(shape)
    }
}

pub trait FixedDimensionKernelBuilder<T> {
    type Params;
    /// Build a fixed size kernel
    fn build() -> Result<Array3<T>, Error>;
    /// Build a fixed size kernel with the given parameters
    fn build_with_params(_p: Self::Params) -> Result<Array3<T>, Error> {
        Self::build()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct GaussianFilter;

/// Builds a Gaussian kernel taking the covariance as a parameter. Covariance
/// is given as 2 values for the x and y variance.
impl<T> KernelBuilder<T> for GaussianFilter
where
    T: Copy + Clone + FromPrimitive + Num,
{
    type Params = [f64; 2];

    fn build<D>(shape: D) -> Result<Array3<T>, Error>
    where
        D: Copy + IntoDimension<Dim = Ix3>,
    {
        // This recommendation was taken from OpenCV 2.4 docs
        let s = shape.clone().into_dimension();
        let sig = 0.3 * (((std::cmp::max(s[0], 1) - 1) as f64) * 0.5 - 1.0) + 0.8;
        Self::build_with_params(shape, [sig, sig])
    }

    fn build_with_params<D>(shape: D, covar: Self::Params) -> Result<Array3<T>, Error>
    where
        D: Copy + IntoDimension<Dim = Ix3>,
    {
        let is_even = |x| x & 1 == 0;
        let s = shape.into_dimension();
        if is_even(s[0]) || is_even(s[1]) || s[0] != s[1] || s[2] == 0 {
            Err(Error::InvalidDimensions)
        } else if covar[0] <= 0.0f64 || covar[1] <= 0.0f64 {
            Err(Error::InvalidParameter)
        } else {
            let centre: isize = (s[0] as isize + 1) / 2 - 1;
            let gauss = |coord, covar| ((coord - centre) as f64).powi(2) / (2.0f64 * covar);

            let mut temp = Array2::from_shape_fn((s[0], s[1]), |(r, c)| {
                f64::exp(-(gauss(r as isize, covar[1]) + gauss(c as isize, covar[0])))
            });

            let sum = temp.sum();

            temp *= 1.0f64 / sum;

            let temp = temp.mapv(|x| T::from_f64(x));

            if temp.iter().any(|x| x.is_none()) {
                Err(Error::NumericError)
            } else {
                let temp = temp.mapv(|x| x.unwrap());
                Ok(Array3::from_shape_fn(shape, |(r, c, _)| temp[[r, c]]))
            }
        }
    }
}

/// The box linear filter is roughly defined as `1/(R*C)*Array2::ones((R, C))`
/// This filter will be a box linear for every colour channel provided
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct BoxLinearFilter;

impl<T> KernelBuilder<T> for BoxLinearFilter
where
    T: Float + Num + NumAssignOps + FromPrimitive,
{
    type Params = bool;
    fn build<D>(shape: D) -> Result<Array3<T>, Error>
    where
        D: Copy + IntoDimension<Dim = Ix3>,
    {
        Self::build_with_params(shape, true)
    }

    fn build_with_params<D>(shape: D, normalise: Self::Params) -> Result<Array3<T>, Error>
    where
        D: Copy + IntoDimension<Dim = Ix3>,
    {
        let shape = shape.into_dimension();
        if shape[0] < 1 || shape[1] < 1 || shape[2] < 1 {
            Err(Error::InvalidDimensions)
        } else if normalise {
            let weight = 1.0f64 / ((shape[0] * shape[1]) as f64);
            match T::from_f64(weight) {
                Some(weight) => Ok(Array3::from_elem(shape, weight)),
                None => Err(Error::NumericError),
            }
        } else {
            Ok(Array3::ones(shape))
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SobelFilter;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Orientation {
    Vertical,
    Horizontal,
}

impl<T> FixedDimensionKernelBuilder<T> for SobelFilter
where
    T: Copy + Clone + Num + Neg<Output = T> + FromPrimitive,
{
    type Params = Orientation;
    /// Build a fixed size kernel
    fn build() -> Result<Array3<T>, Error> {
        // Arbitary decision
        Self::build_with_params(Orientation::Vertical)
    }

    /// Build a fixed size kernel with the given parameters
    fn build_with_params(p: Self::Params) -> Result<Array3<T>, Error> {
        let two = T::from_i8(2).ok_or_else(|| Error::NumericError)?;

        let vert_sobel = arr2(&[
            [-T::one(), T::zero(), T::one()],
            [-two, T::zero(), two],
            [-T::one(), T::zero(), T::one()],
        ]);
        let sobel = match p {
            Orientation::Vertical => vert_sobel,
            Orientation::Horizontal => vert_sobel.t().to_owned(),
        };
        Ok(sobel.insert_axis(Axis(2)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn test_box_linear_filter() {
        let filter: Array3<f64> = BoxLinearFilter::build(Ix3(2, 2, 3)).unwrap();

        assert_eq!(filter, Array3::from_elem((2, 2, 3), 0.25f64));

        let filter: Result<Array3<f64>, Error> = BoxLinearFilter::build(Ix3(0, 2, 3));
        assert!(filter.is_err());
    }

    #[test]
    fn test_sobel_filter() {
        // As sobel works with integer numbers I'm going to ignore the perils of
        // floating point comparisons... for now.
        let filter: Array3<f32> = SobelFilter::build_with_params(Orientation::Vertical).unwrap();

        assert_eq!(
            filter,
            arr3(&[
                [[-1.0f32], [0.0f32], [1.0f32]],
                [[-2.0f32], [0.0f32], [2.0f32]],
                [[-1.0f32], [0.0f32], [1.0f32]]
            ])
        );

        let filter: Array3<f32> = SobelFilter::build_with_params(Orientation::Horizontal).unwrap();

        assert_eq!(
            filter,
            arr3(&[
                [[-1.0f32], [-2.0f32], [-1.0f32]],
                [[0.0f32], [0.0f32], [0.0f32]],
                [[1.0f32], [2.0f32], [1.0f32]]
            ])
        )
    }

    #[test]
    fn test_gaussian_filter() {
        let bad_gauss: Result<Array3<f64>, _> = GaussianFilter::build(Ix3(3, 5, 2));
        assert_eq!(bad_gauss, Err(Error::InvalidDimensions));
        let bad_gauss: Result<Array3<f64>, _> = GaussianFilter::build(Ix3(4, 4, 2));
        assert_eq!(bad_gauss, Err(Error::InvalidDimensions));
        let bad_gauss: Result<Array3<f64>, _> = GaussianFilter::build(Ix3(4, 0, 2));
        assert_eq!(bad_gauss, Err(Error::InvalidDimensions));

        let channels = 2;
        let filter: Array3<f64> =
            GaussianFilter::build_with_params(Ix3(3, 3, channels), [0.3, 0.3]).unwrap();

        assert_eq!(filter.sum().round(), channels as f64);

        let filter: Array3<f64> =
            GaussianFilter::build_with_params(Ix3(3, 3, 1), [0.05, 0.05]).unwrap();

        let filter = filter.mapv(|x| x.round() as u8);
        // Need to do a proper test but this should cover enough
        assert_eq!(
            filter,
            arr3(&[[[0], [0], [0]], [[0], [1], [0]], [[0], [0], [0]]])
        );
    }
}
