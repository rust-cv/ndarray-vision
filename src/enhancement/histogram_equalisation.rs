use crate::core::*;
use ndarray::{prelude::*, DataMut};
use ndarray_stats::{histogram::Grid, HistogramExt};
use num_traits::cast::{FromPrimitive, ToPrimitive};
use num_traits::{Num, NumAssignOps};

/// Extension trait to implement histogram equalisation on other types
pub trait HistogramEqExt<A>
where
    A: Ord,
{
    type Output;
    /// Equalises an image histogram returning a new image.
    /// Grids should be for a 1xN image as the image is flattened during processing
    fn equalise_hist(&self, grid: Grid<A>) -> Self::Output;

    /// Equalises an image histogram inplace
    /// Grids should be for a 1xN image as the image is flattened during processing
    fn equalise_hist_inplace(&mut self, grid: Grid<A>);
}

impl<T, U> HistogramEqExt<T> for ArrayBase<U, Ix3>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive + PixelBound,
{
    type Output = Array<T, Ix3>;

    fn equalise_hist(&self, grid: Grid<T>) -> Self::Output {
        let mut result = self.to_owned();
        result.equalise_hist_inplace(grid);
        result
    }

    fn equalise_hist_inplace(&mut self, grid: Grid<T>) {
        for mut c in self.axis_iter_mut(Axis(2)) {
            // get the histogram
            let flat = Array::from_iter(c.iter()).mapv(|x| *x).insert_axis(Axis(1));
            let hist = flat.histogram(grid.clone());
            // get cdf
            let mut running_total = 0;
            let mut min = 0.0;
            let cdf = hist.counts().mapv(|x| {
                running_total += x;
                if min == 0.0 && running_total > 0 {
                    min = running_total as f32;
                }
                running_total as f32
            });

            // Rescale cdf writing back new values
            let scale = (T::max_pixel() - T::min_pixel())
                .to_f32()
                .unwrap_or_default();
            let denominator = flat.len() as f32 - min;
            c.mapv_inplace(|x| {
                let index = match grid.index_of(&arr1(&[x])) {
                    Some(i) => {
                        if i.is_empty() {
                            0
                        } else {
                            i[0]
                        }
                    }
                    None => 0,
                };
                let mut f_res = ((cdf[index] - min) / denominator) * scale;
                if T::is_integral() {
                    f_res = f_res.round();
                }
                T::from_f32(f_res).unwrap_or_else(T::zero) + T::min_pixel()
            });
        }
    }
}

impl<T, U, C> HistogramEqExt<T> for ImageBase<U, C>
where
    U: DataMut<Elem = T>,
    T: Copy + Clone + Ord + Num + NumAssignOps + ToPrimitive + FromPrimitive + PixelBound,
    C: ColourModel,
{
    type Output = Image<T, C>;

    fn equalise_hist(&self, grid: Grid<T>) -> Self::Output {
        let mut result = self.to_owned();
        result.equalise_hist_inplace(grid);
        result
    }

    fn equalise_hist_inplace(&mut self, grid: Grid<T>) {
        self.data.equalise_hist_inplace(grid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Gray;
    use ndarray_stats::histogram::{Bins, Edges};

    #[test]
    fn hist_eq_test() {
        // test data from wikipedia
        let input_pixels = vec![
            52, 55, 61, 59, 70, 61, 76, 61, 62, 59, 55, 104, 94, 85, 59, 71, 63, 65, 66, 113, 144,
            104, 63, 72, 64, 70, 70, 126, 154, 109, 71, 69, 67, 73, 68, 106, 122, 88, 68, 68, 68,
            79, 60, 79, 77, 66, 58, 75, 69, 85, 64, 58, 55, 61, 65, 83, 70, 87, 69, 68, 65, 73, 78,
            90,
        ];

        let output_pixels = vec![
            0, 12, 53, 32, 146, 53, 174, 53, 57, 32, 12, 227, 219, 202, 32, 154, 65, 85, 93, 239,
            251, 227, 65, 158, 73, 146, 146, 247, 255, 235, 154, 130, 97, 166, 117, 231, 243, 210,
            117, 117, 117, 190, 36, 190, 178, 93, 20, 170, 130, 202, 73, 20, 12, 53, 85, 194, 146,
            206, 130, 117, 85, 166, 182, 215,
        ];

        let input = Image::<u8, Gray>::from_shape_data(8, 8, input_pixels);

        let expected = Image::<u8, Gray>::from_shape_data(8, 8, output_pixels);

        let edges_vec: Vec<u8> = (0..255).collect();
        let grid = Grid::from(vec![Bins::new(Edges::from(edges_vec))]);

        let equalised = input.equalise_hist(grid);

        assert_eq!(expected, equalised);
    }
}
