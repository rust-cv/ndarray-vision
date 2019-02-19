use ndarray::{arr2, Array3, Axis, Ix3};
use num_traits::{cast::FromPrimitive, float::Float, Num, NumAssignOps};
use core::ops::Neg;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Error {
    InvalidDimensions,
    NumericError,
}

pub trait KernelBuilder<T> {
    type Params;
    /// Build a kernel with a given dimension
    fn build(shape: Ix3) -> Result<Array3<T>, Error>;
    /// For kernels with optional parameters use build with params otherwise
    /// appropriate default parameters will be chosen
    fn build_with_params(shape: Ix3, _p: Self::Params) -> Result<Array3<T>, Error> {
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

impl<T> KernelBuilder<T> for GaussianFilter {
    type Params = [[T; 2]; 2];

    fn build(shape: Ix3) -> Result<Array3<T>, Error> {
        unimplemented!()
    }

    fn build_with_params(shape: Ix3, covariance: Self::Params) -> Result<Array3<T>, Error> {
        unimplemented!()
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
    type Params = ();
    fn build(shape: Ix3) -> Result<Array3<T>, Error> {
        if shape[0] < 1 || shape[1] < 1 || shape[2] < 1 {
            Err(Error::InvalidDimensions)
        } else {
            let weight = 1.0f64 / ((shape[0] * shape[1]) as f64);
            match T::from_f64(weight) {
                Some(weight) => Ok(Array3::from_elem(shape, weight)),
                None => Err(Error::NumericError),
            }
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SobelFilter;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum Orientation {
    Vertical, Horizontal
}

impl<T> FixedDimensionKernelBuilder<T> for SobelFilter where T: Copy + Clone + Num + Neg<Output=T> + FromPrimitive {
    type Params = Orientation;
    /// Build a fixed size kernel 
    fn build() -> Result<Array3<T>, Error> {
        // Arbitary decision
        Self::build_with_params(Orientation::Vertical)
    }

    /// Build a fixed size kernel with the given parameters
    fn build_with_params(p: Self::Params) -> Result<Array3<T>, Error> {
        let two = T::from_i8(2).ok_or_else(|| Error::NumericError)?;
        
        let vert_sobel = arr2(&[[-T::one(),T::zero(), T::one()],
                               [-two, T::zero(), two],
                               [-T::one(), T::zero(), T::one()]]);
        let sobel = match p {
            Orientation::Vertical => {
                vert_sobel
            },
            Orientation::Horizontal => {
                vert_sobel.t().to_owned()
            }
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
        let filter: Array3<f32> = SobelFilter::build_with_params(Orientation::Vertical).unwrap();

        assert_eq!(filter, arr3(&[[[-1.0f32], [0.0f32], [1.0f32]],
                                  [[-2.0f32], [0.0f32], [2.0f32]],
                                  [[-1.0f32], [0.0f32], [1.0f32]]]));
        
        let filter: Array3<f32> = SobelFilter::build_with_params(Orientation::Horizontal).unwrap();

        assert_eq!(filter, arr3(&[[[-1.0f32], [-2.0f32], [-1.0f32]],
                                  [[0.0f32], [0.0f32], [0.0f32]],
                                  [[1.0f32], [2.0f32], [1.0f32]]]))
    }
}
