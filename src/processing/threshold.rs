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
use std::iter::FromIterator;
use std::marker::PhantomData;

// Development
#[cfg(test)]
use assert_approx_eq::assert_approx_eq;
#[cfg(test)]
use noisy_float::types::n64;

/// Runs the Otsu Thresholding algorithm on a type T
pub trait ThresholdOtsuExt<T> {
    /// Output type, this is different as Otsu outputs a binary image
    type Output;

    /// Run the Otsu threshold detection algorithm with the
    /// given parameters. Due to Otsu being specified as working
    /// on greyscale images all current implementations
    /// assume a single channel image returning an error otherwise.
    fn threshold_otsu(&self) -> Result<Self::Output, Error>;
}

/// Runs the Mean Thresholding algorithm on a type T
pub trait ThresholdMeanExt<T> {
    /// Output type, this is different as Otsu outputs a binary image
    type Output;

    /// Run the Otsu threshold detection algorithm with the
    /// given parameters. Due to Otsu being specified as working
    /// on greyscale images all current implementations
    /// assume a single channel image returning an error otherwise.
    fn threshold_mean(&self) -> Result<Self::Output, Error>;
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
            let mask = apply_threshold(self, value);
            Ok(mask)
        }
    }
}

///
/// Calculates Otsu's threshold
/// Works per channel, but currently
/// assumes grayscale (see the error above if number of channels is > 1
/// i.e. single channel; otherwise we need to output all 3 threshold values).
/// Todo: Add optional nbins
///
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
            let mask = apply_threshold(self, value);
            Ok(mask)
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

fn apply_threshold<T, U>(data: &ArrayBase<U, Ix3>, threshold: f64) -> Array3<bool>
where
    U: Data<Elem = T>,
    T: Copy + Clone + Num + NumAssignOps + ToPrimitive + FromPrimitive,
{
    let result = data.mapv(|x| x.to_f64().unwrap() >= threshold);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

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

        let result = apply_threshold(&data, 0.5);

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
