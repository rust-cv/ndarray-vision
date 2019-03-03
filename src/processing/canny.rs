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
        if self.shape()[2] > 1 {
            Err(Error::ChannelDimensionMismatch) 
        } else {
            let t1 = params.t1;
            let t2 = params.t2;
            // apply blur 
            let blurred = self.conv2d(params.blur.view())?; 
            let (mut mag, rot) = blurred.full_sobel()?;
            mag.mapv_inplace(|x| if x >= t1 { x  } else { T::zero() });
            
            let mag = non_maxima_supression(mag, rot.view());
            
            Ok(link_edges(mag, t1, t2))
        }
    }
}


fn non_maxima_supression<T>(magnitudes: Array3<T>, rotations: ArrayView3<T>) -> Array3<T> 
where 
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps
{
    let row_size = magnitudes.shape()[0] as isize;
    let column_size = magnitudes.shape()[1] as isize;

    let get_neighbours = |r, c, dr, dc| {
        if (r == 0 && dr < 0) || (r==(row_size-1) && dr > 0) {
            T::zero()
        } else if (c==0 && dc < 0) || (c==(column_size-1) && dr > 0) {
            T::zero()
        } else {
            magnitudes[[(r+dr) as usize, (c+dc) as usize, 0]]
        }
    };

    let mut result = magnitudes.clone();

    for (i, mut row) in result.outer_iter_mut().enumerate() {
        let i = i as isize;
        for (j, mut col) in row.outer_iter_mut().enumerate() {
            let  mut dir = rotations[[i as usize, j, 0]]
                .to_degrees()
                .to_f64()
                .unwrap_or_else(|| 0.0);

            let j = j as isize;
            if dir >= 180.0 {
                dir -= 180.0;
            } else if dir < 0.0 {
                dir += 180.0;
            }
            // Now get neighbour values and suppress col if not a maxima
            let (a ,b) = if dir < 45.0 {
                (get_neighbours(i, j, 0, -1), get_neighbours(i, j, 0, 1))
            } else if dir < 90.0 {
                (get_neighbours(i, j, -1, -1), get_neighbours(i, j, 1, 1))
            } else if dir < 135.0 {
                (get_neighbours(i, j, -1, 0), get_neighbours(i, j, 1, 0))
            } else {
                (get_neighbours(i, j, -1, 1), get_neighbours(i, j, 1, -1))
            };

            if a > col[[0]] || b > col[[0]] {
                col.fill(T::zero());
            }
        }
    }
    result
}

fn link_edges<T>(magnitudes: Array3<T>, lower: T, upper: T) -> Array3<bool> 
where 
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps
{
    let mut result = magnitudes.mapv(|x| x >= upper);
    let rows = result.shape()[0];
    let cols = result.shape()[1];
    
    for r in 0..rows {
        for c in 0..cols {
            // If it is a strong edge check if neighbours are weak and add them
            if result[[r, c, 0]]  {
                if r > 0 && magnitudes[[r-1, c, 0]] >= lower {
                    result[[r-1, c, 0]] = true;
                    if c > 0 && magnitudes[[r-1, c-1, 0]] >= lower {
                        result[[r-1, c-1, 0]] = true;
                    }
                    if c < cols -1 && magnitudes[[r-1, c+1, 0]] >= lower {
                        result[[r-1, c+1, 0]] = true;
                    }
                }
                if r < rows -1 && magnitudes[[r+1, c, 0]] >= lower {
                    result[[r+1, c, 0]] = true;
                    if c > 0 && magnitudes[[r+1, c-1, 0]] >= lower {
                        result[[r+1, c-1, 0]] = true;
                    }
                    if c < cols -1 && magnitudes[[r+1, c+1, 0]] >= lower {
                        result[[r+1, c+1, 0]] = true;
                    }
                }
                if c > 0 && magnitudes[[r, c-1, 0]] >= lower {
                    result[[r, c-1, 0]] = true;
                }
                if c < cols -1 && magnitudes[[r, c+1, 0]] >= lower {
                    result[[r, c+1, 0]] = true;
                }
            }   
        }
    }

    result
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
        D: Copy + IntoDimension<Dim = Ix2>,
    {
        let shape = shape.into_dimension();
        let shape = (shape[0], shape[1], 1);
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



