use ndarray::{Array3, Ix3};
use num_traits::{cast::FromPrimitive, float::Float, Num, NumAssignOps};

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum Error {
    InvalidDimensions,
    NumericError,
}

pub trait KernelBuilder<T> {
    fn build(shape: Ix3) -> Result<Array3<T>, Error>;
}

/// The box linear filter is roughly defined as `1/(R*C)*Array2::ones((R, C))`
/// This filter will be a box linear for every colour channel provided
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct BoxLinearFilter;

impl<T> KernelBuilder<T> for BoxLinearFilter
where
    T: Float + Num + NumAssignOps + FromPrimitive,
{
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_linear_filter() {
        let filter: Array3<f64> = BoxLinearFilter::build(Ix3(2, 2, 3)).unwrap();

        assert_eq!(filter, Array3::from_elem((2, 2, 3), 0.25f64));

        let filter: Result<Array3<f64>, Error> = BoxLinearFilter::build(Ix3(0, 2, 3));
        assert!(filter.is_err());
    }
}
