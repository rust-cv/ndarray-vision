use crate::core::PixelBound;
use crate::core::{ColourModel, Image, ImageBase};
use crate::processing::*;
use ndarray::{prelude::*, Data};
use ndarray_stats::histogram::{Bins, Edges, Grid};
use ndarray_stats::HistogramExt;
use ndarray_stats::QuantileExt;
use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use num_traits::{Num, NumAssignOps};
use std::marker::PhantomData;

/// Runs the Otsu thresholding algorithm on a type `T`.
pub trait ThresholdOtsuExt<T> {
    /// The Otsu thresholding output is a binary image.
    type Output;

    /// Run the Otsu threshold algorithm.
    ///
    /// Due to Otsu threshold algorithm specifying a greyscale image, all
    /// current implementations assume a single channel image; otherwise, an
    /// error is returned.
    ///
    /// # Errors
    ///
    /// Returns a `ChannelDimensionMismatch` error if more than one channel
    /// exists.
    fn threshold_otsu(&self) -> Result<Self::Output, Error>;
}

/// Runs the Mean thresholding algorithm on a type `T`.
pub trait ThresholdMeanExt<T> {
    /// The Mean thresholding output is a binary image.
    type Output;

    /// Run the Mean threshold algorithm.
    ///
    /// This assumes the image is a single channel image, i.e., a greyscale
    /// image; otherwise, an error is returned.
    ///
    /// # Errors
    ///
    /// Returns a `ChannelDimensionMismatch` error if more than one channel
    /// exists.
    fn threshold_mean(&self) -> Result<Self::Output, Error>;
}

/// Applies an upper and lower limit threshold on a type `T`.
pub trait ThresholdApplyExt<T> {
    /// The output is a binary image.
    type Output;

    /// Apply the threshold with the given limits.
    ///
    /// An image is segmented into background and foreground
    /// elements, where any pixel value within the limits are considered
    /// foreground elements and any pixels with a value outside the limits are
    /// considered part of the background. The upper and lower limits are
    /// inclusive.
    ///
    /// If only a lower limit threshold is to be applied, the `f64::INFINITY`
    /// value can be used for the upper limit.
    ///
    /// # Errors
    ///
    /// The current implementation assumes a single channel image, i.e.,
    /// greyscale image. Thus, if more than one channel is present, then
    /// a `ChannelDimensionMismatch` error occurs.
    ///
    /// An `InvalidParameter` error occurs if the `lower` limit is greater than
    /// the `upper` limit.
    fn threshold_apply(&self, lower: f64, upper: f64) -> Result<Self::Output, Error>;
}

impl<T, U, C> ThresholdOtsuExt<T> for ImageBase<U, C>
where
    U: Data<Elem = T>,
    Image<U, C>: Clone,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive + PixelBound,
    C: ColourModel,
{
    type Output = Image<bool, C>;

    fn threshold_otsu(&self) -> Result<Self::Output, Error> {
        let data = self.data.threshold_otsu()?;
        Ok(Self::Output {
            data,
            model: PhantomData,
        })
    }
}

impl<T, U> ThresholdOtsuExt<T> for ArrayBase<U, Ix3>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    type Output = Array3<bool>;

    fn threshold_otsu(&self) -> Result<Self::Output, Error> {
        if self.shape()[2] > 1 {
            Err(Error::ChannelDimensionMismatch)
        } else {
            let value = calculate_threshold_otsu(self)?;
            self.threshold_apply(value, f64::INFINITY)
        }
    }
}

/// Calculates Otsu's threshold.
///
/// Works per channel, but currently assumes greyscale.
///
/// See the Errors section for the `ThresholdOtsuExt` trait if the number of
/// channels is greater than one (1), i.e., single channel; otherwise, we would
/// need to output all three threshold values.
///
/// TODO: Add optional nbins
fn calculate_threshold_otsu<T, U>(mat: &ArrayBase<U, Ix3>) -> Result<f64, Error>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    let mut threshold = 0.0;
    let n_bins = 255;
    for c in mat.axis_iter(Axis(2)) {
        let scale_factor = (n_bins) as f64 / (c.max().unwrap().to_f64().unwrap());
        let edges_vec: Vec<u8> = (0..n_bins).collect();
        let grid = Grid::from(vec![Bins::new(Edges::from(edges_vec))]);

        // get the histogram
        let flat = Array::from_iter(c.iter()).insert_axis(Axis(1));
        let flat2 = flat.mapv(|x| ((*x).to_f64().unwrap() * scale_factor).to_u8().unwrap());
        let hist = flat2.histogram(grid);
        // Straight out of wikipedia:
        let counts = hist.counts();
        let total = counts.sum().to_f64().unwrap();
        let counts = Array::from_iter(counts.iter());
        // NOTE: Could use the cdf generation for skimage-esque implementation
        // which entails a cumulative sum of the standard histogram
        let mut sum_b = 0.0;
        let mut weight_b = 0.0;
        let mut maximum = 0.0;
        let mut level = 0.0;
        let mut sum_intensity = 0.0;
        for (index, count) in counts.indexed_iter() {
            sum_intensity += (index as f64) * (*count).to_f64().unwrap();
        }
        for (index, count) in counts.indexed_iter() {
            weight_b = weight_b + count.to_f64().unwrap();
            sum_b = sum_b + (index as f64) * count.to_f64().unwrap();
            let weight_f = total - weight_b;
            if (weight_b > 0.0) && (weight_f > 0.0) {
                let mean_f = (sum_intensity - sum_b) / weight_f;
                let val = weight_b
                    * weight_f
                    * ((sum_b / weight_b) - mean_f)
                    * ((sum_b / weight_b) - mean_f);
                if val > maximum {
                    level = 1.0 + (index as f64);
                    maximum = val;
                }
            }
        }
        threshold = level as f64 / scale_factor;
    }
    Ok(threshold)
}

impl<T, U, C> ThresholdMeanExt<T> for ImageBase<U, C>
where
    U: Data<Elem = T>,
    Image<U, C>: Clone,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive + PixelBound,
    C: ColourModel,
{
    type Output = Image<bool, C>;

    fn threshold_mean(&self) -> Result<Self::Output, Error> {
        let data = self.data.threshold_mean()?;
        Ok(Self::Output {
            data,
            model: PhantomData,
        })
    }
}

impl<T, U> ThresholdMeanExt<T> for ArrayBase<U, Ix3>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    type Output = Array3<bool>;

    fn threshold_mean(&self) -> Result<Self::Output, Error> {
        if self.shape()[2] > 1 {
            Err(Error::ChannelDimensionMismatch)
        } else {
            let value = calculate_threshold_mean(self)?;
            self.threshold_apply(value, f64::INFINITY)
        }
    }
}

fn calculate_threshold_mean<T, U>(array: &ArrayBase<U, Ix3>) -> Result<f64, Error>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    Ok(array.sum().to_f64().unwrap() / array.len() as f64)
}

impl<T, U, C> ThresholdApplyExt<T> for ImageBase<U, C>
where
    U: Data<Elem = T>,
    Image<U, C>: Clone,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive + PixelBound,
    C: ColourModel,
{
    type Output = Image<bool, C>;

    fn threshold_apply(&self, lower: f64, upper: f64) -> Result<Self::Output, Error> {
        let data = self.data.threshold_apply(lower, upper)?;
        Ok(Self::Output {
            data,
            model: PhantomData,
        })
    }
}

impl<T, U> ThresholdApplyExt<T> for ArrayBase<U, Ix3>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    type Output = Array3<bool>;

    fn threshold_apply(&self, lower: f64, upper: f64) -> Result<Self::Output, Error> {
        if self.shape()[2] > 1 {
            Err(Error::ChannelDimensionMismatch)
        } else if lower > upper {
            Err(Error::InvalidParameter)
        } else {
            Ok(apply_threshold(self, lower, upper))
        }
    }
}

fn apply_threshold<T, U>(data: &ArrayBase<U, Ix3>, lower: f64, upper: f64) -> Array3<bool>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    data.mapv(|x| x.to_f64().unwrap() >= lower && x.to_f64().unwrap() <= upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use ndarray::arr3;
    use noisy_float::types::n64;

    #[test]
    fn threshold_apply_threshold() {
        let data = arr3(&[
            [[0.2], [0.4], [0.0]],
            [[0.7], [0.5], [0.8]],
            [[0.1], [0.6], [0.0]],
        ]);

        let expected = arr3(&[
            [[false], [false], [false]],
            [[true], [true], [true]],
            [[false], [true], [false]],
        ]);

        let result = apply_threshold(&data, 0.5, f64::INFINITY);

        assert_eq!(result, expected);
    }

    #[test]
    fn threshold_apply_threshold_range() {
        let data = arr3(&[
            [[0.2], [0.4], [0.0]],
            [[0.7], [0.5], [0.8]],
            [[0.1], [0.6], [0.0]],
        ]);
        let expected = arr3(&[
            [[false], [true], [false]],
            [[true], [true], [false]],
            [[false], [true], [false]],
        ]);

        let result = apply_threshold(&data, 0.25, 0.75);

        assert_eq!(result, expected);
    }

    #[test]
    fn threshold_calculate_threshold_otsu_ints() {
        let data = arr3(&[[[2], [4], [0]], [[7], [5], [8]], [[1], [6], [0]]]);
        let result = calculate_threshold_otsu(&data).unwrap();
        println!("Done {}", result);

        // Calculated using Python's skimage.filters.threshold_otsu
        // on int input array. Float array returns 2.0156...
        let expected = 2.0;

        assert_approx_eq!(result, expected, 5e-1);
    }

    #[test]
    fn threshold_calculate_threshold_otsu_floats() {
        let data = arr3(&[
            [[n64(2.0)], [n64(4.0)], [n64(0.0)]],
            [[n64(7.0)], [n64(5.0)], [n64(8.0)]],
            [[n64(1.0)], [n64(6.0)], [n64(0.0)]],
        ]);

        let result = calculate_threshold_otsu(&data).unwrap();

        // Calculated using Python's skimage.filters.threshold_otsu
        // on int input array. Float array returns 2.0156...
        let expected = 2.0156;

        assert_approx_eq!(result, expected, 5e-1);
    }

    #[test]
    fn threshold_calculate_threshold_mean_ints() {
        let data = arr3(&[[[4], [4], [4]], [[5], [5], [5]], [[6], [6], [6]]]);

        let result = calculate_threshold_mean(&data).unwrap();
        let expected = 5.0;

        assert_approx_eq!(result, expected, 1e-16);
    }

    #[test]
    fn threshold_calculate_threshold_mean_floats() {
        let data = arr3(&[
            [[4.0], [4.0], [4.0]],
            [[5.0], [5.0], [5.0]],
            [[6.0], [6.0], [6.0]],
        ]);

        let result = calculate_threshold_mean(&data).unwrap();
        let expected = 5.0;

        assert_approx_eq!(result, expected, 1e-16);
    }
}
