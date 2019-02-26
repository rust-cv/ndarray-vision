use crate::core::*;
use ndarray::prelude::*;
use ndarray_stats::{HistogramExt, histogram::{Grid, Histogram}};
use num_traits::{Num, NumAssignOps};

pub trait HistogramEqExt<A> where A: Ord {
    fn equalise_hist(self, grid: Grid<A>) -> Self;

    fn equalise_hist_inplace(&mut self, grid: Grid<A>);
}


impl<T> HistogramEqExt<T> for Array3<T>
where
    T: Copy + Clone + Ord + Num + NumAssignOps 
{
    fn equalise_hist(self, grid: Grid<T>) -> Self {
        let mut result = self.clone();
        result.equalise_hist_inplace(grid);
        result
    }

    fn equalise_hist_inplace(&mut self, grid: Grid<T>) {
        for c in self.axis_iter_mut(Axis(2)) {
            // get the histogram
            
            // get cdf 

            // Rescale cdf writing back new values
        }
        unimplemented!()
    }
}


impl<T, C> HistogramEqExt<T> for Image<T, C> 
where 
    Image<T, C>: Clone,
    T: Copy + Clone + Ord + Num + NumAssignOps,
    C: ColourModel,
{
    fn equalise_hist(self, grid: Grid<T>) -> Self {
        let mut result = self.clone();
        result.equalise_hist_inplace(grid);
        result
    }

    fn equalise_hist_inplace(&mut self, grid: Grid<T>) {
        self.data.equalise_hist_inplace(grid);
    }
}
