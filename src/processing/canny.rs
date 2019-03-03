use crate::processing::kernels::*;
use ndarray::IntoDimension;
use ndarray::prelude::*;
use num_traits::{Num, real::Real, cast::FromPrimitive};

pub trait CannyEdgeDetectorExt<T> {
    type Output;

   fn canny_edge_detector(params: CannyParameters<T>) -> Self::Output;
}


impl<T> CannyEdgeDetectorExt<T> for Array3<T> 
where 
    T: Copy + Clone + FromPrimitive + Real + Num
{
    type Output = Array3<bool>;

   fn canny_edge_detector(params: CannyParameters<T>) -> Self::Output {
        unimplemented!()
   }
}


#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct CannyBuilder<T> {
    blur: Option<Array3<T>>,
    t1: Option<T>,
    t2: Option<T>,
}

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct CannyParameters<T> {
    blur: Array3<T>,
    t1: T,
    t2: T,
}


impl<T> CannyBuilder<T>
where
    T: Copy + Clone + FromPrimitive + Real + Num
{
    pub fn lower_threshold(self, t1: T) -> Self {
        Self {
            blur: self.blur,
            t1: Some(t1),
            t2: self.t2,
        }
    }

    pub fn upper_threshold(self, t2: T) -> Self {
        Self {
            blur: self.blur,
            t1: self.t1,
            t2: Some(t2),
        }
    }

    pub fn blur<D>(self, shape: D, covariance: [f64;2]) -> Self 
    where
        D: Copy + IntoDimension<Dim = Ix3>,
    {
        if let Ok(blur) = GaussianFilter::build_with_params(shape, covariance) {
            Self {
                blur: Some(blur),
                t1: self.t1,
                t2: self.t2,
            }
        } else {
            self
        }
    }

    pub fn build(self) -> CannyParameters<T> {
        let blur = match self.blur {
            Some(b) => b,
            None => GaussianFilter::build_with_params((5, 5, 1), [2.0, 2.0]).unwrap(),
        };
        let t1 = match self.t1 {
            Some(t) => t,
            None => T::from_f64(0.3).unwrap(),
        };
        let t2 = match self.t2 {
            Some(t) => t,
            None => T::from_f64(0.7).unwrap(),
        };
        CannyParameters {
            blur,
            t1,
            t2
        }
    }
}



