use crate::processing::*;
use ndarray::IntoDimension;
use ndarray::prelude::*;
use num_traits::{Num, NumAssignOps, real::Real, cast::FromPrimitive};

pub trait CannyEdgeDetectorExt<T> {
    type Output;

   fn canny_edge_detector(&self, params: CannyParameters<T>) -> Result<Self::Output, Error>;
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


impl<T> CannyEdgeDetectorExt<T> for Array3<T> 
where 
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps
{
    type Output = Array3<bool>;

    fn canny_edge_detector(&self, params: CannyParameters<T>) -> Result<Self::Output, Error> {
        let t1 = params.t1;
        let t2 = params.t2;
        // First check blur is right width and if not expand
        let blur = if params.blur.shape()[2] == self.shape()[2] {
            params.blur
        } else if params.blur.shape()[2] == 1 {
            Array::from_shape_fn(params.blur.dim(), |(i, j, _)| params.blur[[i, j, 0]])
        } else {
            return Err(Error::ChannelDimensionMismatch);   
        };
        // apply blur 
        let blurred = self.conv2d(blur.view())?; 
        let (mut mag, rot) = blurred.full_sobel()?;
        mag.mapv_inplace(|x| if x >= t1 { x  } else { T::zero() });
        
        non_maxima_supression(mag.view_mut(), rot.view());
        
        Ok(link_edges(mag, t1, t2))
    }
}


fn non_maxima_supression<T>(magnitudes: ArrayViewMut3<T>, rotations: ArrayView3<T>) 
where 
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps
{
    unimplemented!()
}

fn link_edges<T>(magnitudes: Array3<T>, lower: T, upper: T) -> Array3<bool> 
where 
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps
{
    unimplemented!()
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
        let mut t1 = match self.t1 {
            Some(t) => t,
            None => T::from_f64(0.3).unwrap(),
        };
        let mut t2 = match self.t2 {
            Some(t) => t,
            None => T::from_f64(0.7).unwrap(),
        };
        if t2 < t1 {
            let temp = t1;
            t1 = t2;
            t2 = temp;
        }
        CannyParameters {
            blur,
            t1,
            t2
        }
    }
}



