use crate::core::{ColourModel, Image, ImageBase};
use crate::processing::*;
use ndarray::prelude::*;
use ndarray::{DataMut, IntoDimension};
use num_traits::{cast::FromPrimitive, real::Real, Num, NumAssignOps};
use std::collections::HashSet;
use std::marker::PhantomData;

/// Runs the Canny Edge Detector algorithm on a type T
pub trait CannyEdgeDetectorExt<T> {
    /// Output type, this is different as canny outputs a binary image
    type Output;

    /// Run the edge detection algorithm with the given parameters. Due to Canny
    /// being specified as working on greyscale images all current implementations
    /// assume a single channel image returning an error otherwise.
    fn canny_edge_detector(&self, params: CannyParameters<T>) -> Result<Self::Output, Error>;
}

/// Builder to construct the Canny parameters, if a parameter is not selected then
/// a sensible default is chosen
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct CannyBuilder<T> {
    blur: Option<Array3<T>>,
    t1: Option<T>,
    t2: Option<T>,
}

/// Parameters for the Canny Edge Detector
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct CannyParameters<T> {
    /// By default this library uses a Gaussian blur, although other kernels can
    /// be substituted
    pub blur: Array3<T>,
    /// Lower threshold for weak edges used during the hystersis based edge linking
    pub t1: T,
    /// Upper threshold defining a strong edge
    pub t2: T,
}

impl<T, U, C> CannyEdgeDetectorExt<T> for ImageBase<U, C>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps,
    C: ColourModel,
{
    type Output = Image<bool, C>;

    fn canny_edge_detector(&self, params: CannyParameters<T>) -> Result<Self::Output, Error> {
        let data = self.data.canny_edge_detector(params)?;
        Ok(Self::Output {
            data,
            model: PhantomData,
        })
    }
}

impl<T, U> CannyEdgeDetectorExt<T> for ArrayBase<U, Ix3>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps,
{
    type Output = Array3<bool>;

    fn canny_edge_detector(&self, params: CannyParameters<T>) -> Result<Self::Output, Error> {
        if self.shape()[2] > 1 {
            Err(Error::ChannelDimensionMismatch)
        } else {
            // apply blur
            let blurred = self.conv2d(params.blur.view())?;
            let (mag, rot) = blurred.full_sobel()?;

            let mag = non_maxima_supression(mag, rot.view());

            Ok(link_edges(mag, params.t1, params.t2))
        }
    }
}

fn non_maxima_supression<T>(magnitudes: Array3<T>, rotations: ArrayView3<T>) -> Array3<T>
where
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps,
{
    let row_size = magnitudes.shape()[0] as isize;
    let column_size = magnitudes.shape()[1] as isize;

    let get_neighbours = |r, c, dr, dc| {
        if (r == 0 && dr < 0) || (r == (row_size - 1) && dr > 0) {
            T::zero()
        } else if (c == 0 && dc < 0) || (c == (column_size - 1) && dc > 0) {
            T::zero()
        } else {
            magnitudes[[(r + dr) as usize, (c + dc) as usize, 0]]
        }
    };

    let mut result = magnitudes.clone();

    for (i, mut row) in result.outer_iter_mut().enumerate() {
        let i = i as isize;
        for (j, mut col) in row.outer_iter_mut().enumerate() {
            let mut dir = rotations[[i as usize, j, 0]]
                .to_degrees()
                .to_f64()
                .unwrap_or(0.0);

            let j = j as isize;
            if dir >= 180.0 {
                dir -= 180.0;
            } else if dir < 0.0 {
                dir += 180.0;
            }
            // Now get neighbour values and suppress col if not a maxima
            let (a, b) = if dir < 45.0 {
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

fn get_candidates(
    coord: (usize, usize),
    bounds: (usize, usize),
    closed_set: &HashSet<[usize; 2]>,
) -> Vec<[usize; 2]> {
    let mut result = Vec::new();
    let (r, c) = coord;
    let (rows, cols) = bounds;

    if r > 0 {
        if c > 0 && !closed_set.contains(&[r - 1, c + 1]) {
            result.push([r - 1, c - 1]);
        }
        if c < cols - 1 && !closed_set.contains(&[r - 1, c + 1]) {
            result.push([r - 1, c + 1]);
        }
        if !closed_set.contains(&[r - 1, c]) {
            result.push([r - 1, c]);
        }
    }
    if r < rows - 1 {
        if c > 0 && !closed_set.contains(&[r + 1, c - 1]) {
            result.push([r + 1, c - 1]);
        }
        if c < cols - 1 && !closed_set.contains(&[r + 1, c + 1]) {
            result.push([r + 1, c + 1]);
        }
        if !closed_set.contains(&[r + 1, c]) {
            result.push([r + 1, c]);
        }
    }
    result
}

fn link_edges<T>(magnitudes: Array3<T>, lower: T, upper: T) -> Array3<bool>
where
    T: Copy + Clone + FromPrimitive + Real + Num + NumAssignOps,
{
    let magnitudes = magnitudes.mapv(|x| if x >= lower { x } else { T::zero() });
    let mut result = magnitudes.mapv(|x| x >= upper);
    let mut visited = HashSet::new();

    let rows = result.shape()[0];
    let cols = result.shape()[1];

    for r in 0..rows {
        for c in 0..cols {
            // If it is a strong edge check if neighbours are weak and add them
            if result[[r, c, 0]] {
                visited.insert([r, c]);
                let mut buffer = get_candidates((r, c), (rows, cols), &visited);

                while let Some(cand) = buffer.pop() {
                    let coord3 = [cand[0], cand[1], 0];
                    if magnitudes[coord3] > lower {
                        visited.insert(cand);
                        result[coord3] = true;

                        let temp = get_candidates((cand[0], cand[1]), (rows, cols), &visited);
                        buffer.extend_from_slice(temp.as_slice());
                    }
                }
            }
        }
    }
    result
}

impl<T> CannyBuilder<T>
where
    T: Copy + Clone + FromPrimitive + Real + Num,
{
    /// Creates a new Builder with no parameters selected
    pub fn new() -> Self {
        Self {
            blur: None,
            t1: None,
            t2: None,
        }
    }

    /// Sets the lower threshold for the parameters returning a new builder
    pub fn lower_threshold(self, t1: T) -> Self {
        Self {
            blur: self.blur,
            t1: Some(t1),
            t2: self.t2,
        }
    }

    /// Sets the upper threshold for the parameters returning a new builder
    pub fn upper_threshold(self, t2: T) -> Self {
        Self {
            blur: self.blur,
            t1: self.t1,
            t2: Some(t2),
        }
    }

    /// Given the shape and covariance matrix constructs a Gaussian blur to be
    /// used with the Canny Edge Detector
    pub fn blur<D>(self, shape: D, covariance: [f64; 2]) -> Self
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

    /// Creates the Canny parameters to be used with sensible defaults for unspecified
    /// parameters. This method also rearranges the upper and lower threshold to
    /// ensure that the relationship `t1 <= t2` is maintained.
    ///
    /// Defaults are: a lower threshold of 0.3, upper threshold of 0.7 and a 5x5
    /// Gaussian blur with a horizontal and vertical variances of 2.0.
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
            std::mem::swap(&mut t1, &mut t2);
        }
        CannyParameters { blur, t1, t2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn canny_builder() {
        let builder = CannyBuilder::<f64>::new()
            .lower_threshold(0.75)
            .upper_threshold(0.25);

        assert_eq!(builder.t1, Some(0.75));
        assert_eq!(builder.t2, Some(0.25));
        assert_eq!(builder.blur, None);

        let result = builder.clone().build();

        assert_eq!(result.t1, 0.25);
        assert_eq!(result.t2, 0.75);

        let builder2 = builder.blur((3, 3), [0.2, 0.2]);

        assert_eq!(builder2.t1, Some(0.75));
        assert_eq!(builder2.t2, Some(0.25));
        assert!(builder2.blur.is_some());
        let gauss = builder2.blur.unwrap();
        assert_eq!(gauss.shape(), [3, 3, 1]);
    }

    #[test]
    fn canny_thresholding() {
        let magnitudes = arr3(&[
            [[0.2], [0.4], [0.0]],
            [[0.7], [0.5], [0.8]],
            [[0.1], [0.6], [0.0]],
        ]);

        let expected = arr3(&[
            [[false], [false], [false]],
            [[true], [true], [true]],
            [[false], [true], [false]],
        ]);

        let result = link_edges(magnitudes, 0.4, 0.69);

        assert_eq!(result, expected);
    }
}
