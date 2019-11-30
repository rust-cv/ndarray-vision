use crate::core::*;
use ndarray::{prelude::*, Data, DataMut, Zip};

pub trait MorphologyExt {
    type Output;

    fn erode(&self, kernel: ArrayView2<bool>) -> Self::Output;

    fn erode_inplace(&mut self, kernel: ArrayView2<bool>);

    fn dilate(&self, kernel: ArrayView2<bool>) -> Self::Output;

    fn dilate_inplace(&mut self, kernel: ArrayView2<bool>);

    fn union(&self, other: &Self) -> Self::Output;

    fn union_inplace(&mut self, other: &Self);

    fn intersect(&self, other: &Self) -> Self::Output;

    fn intersect_inplace(&mut self, other: &Self);
}

impl<U> MorphologyExt for ArrayBase<U, Ix3>
where
    U: Data<Elem = bool> + DataMut<Elem = bool>,
{
    type Output = Array<bool, Ix3>;

    fn erode(&self, kernel: ArrayView2<bool>) -> Self::Output {
        let sh = kernel.shape();
        let (ro, co) = kernel_centre(sh[0], sh[1]);
        let mut result = Self::Output::from_elem(self.dim(), false);
        if self.shape()[0] >= sh[0] && self.shape()[1] >= sh[1] {
            Zip::indexed(self.slice(s![.., .., 0]).windows(kernel.dim())).apply(
                |(i, j), window| {
                    result[[i + ro, j + co, 0]] = (&kernel & &window) == kernel;
                },
            );
        }
        result
    }

    fn erode_inplace(&mut self, kernel: ArrayView2<bool>) {
        self.assign(&self.erode(kernel));
    }

    fn dilate(&self, kernel: ArrayView2<bool>) -> Self::Output {
        let sh = kernel.shape();
        let (ro, co) = kernel_centre(sh[0], sh[1]);
        let mut result = Self::Output::from_elem(self.dim(), false);
        if self.shape()[0] >= sh[0] && self.shape()[1] >= sh[1] {
            Zip::indexed(self.slice(s![.., .., 0]).windows(kernel.dim())).apply(
                |(i, j), window| {
                    result[[i + ro, j + co, 0]] = (&kernel & &window).iter().any(|x| *x);
                },
            );
        }
        result
    }

    fn dilate_inplace(&mut self, kernel: ArrayView2<bool>) {
        self.assign(&self.dilate(kernel));
    }

    fn union(&self, other: &Self) -> Self::Output {
        self | other
    }

    fn union_inplace(&mut self, other: &Self) {
        *self |= other;
    }

    fn intersect(&self, other: &Self) -> Self::Output {
        self & other
    }

    fn intersect_inplace(&mut self, other: &Self) {
        *self &= other;
    }
}

impl<U, C> MorphologyExt for ImageBase<U, C>
where
    U: Data<Elem = bool> + DataMut<Elem = bool>,
    C: ColourModel,
{
    type Output = Image<bool, C>;

    fn erode(&self, kernel: ArrayView2<bool>) -> Self::Output {
        Self::Output::from_array(self.data.erode(kernel))
    }

    fn erode_inplace(&mut self, kernel: ArrayView2<bool>) {
        self.data.erode_inplace(kernel);
    }

    fn dilate(&self, kernel: ArrayView2<bool>) -> Self::Output {
        Self::Output::from_array(self.data.dilate(kernel))
    }

    fn dilate_inplace(&mut self, kernel: ArrayView2<bool>) {
        self.data.dilate_inplace(kernel);
    }

    fn union(&self, other: &Self) -> Self::Output {
        Self::Output::from_array(self.data.union(&other.data))
    }

    fn union_inplace(&mut self, other: &Self) {
        self.data.union_inplace(&other.data);
    }

    fn intersect(&self, other: &Self) -> Self::Output {
        Self::Output::from_array(self.data.intersect(&other.data))
    }

    fn intersect_inplace(&mut self, other: &Self) {
        self.data.intersect_inplace(&other.data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn simple_dilation() {
        let pix_in = vec![
            false, false, false, false, false, false, false, false, false, false, false, false,
            true, false, false, false, false, false, false, false, false, false, false, false,
            false,
        ];
        let pix_out = vec![
            false, false, false, false, false, false, true, true, true, false, false, true, true,
            true, false, false, true, true, true, false, false, false, false, false, false,
        ];

        let kern = arr2(&[[true, true, true], [true, true, true], [true, true, true]]);

        let mut input = Image::<bool, Gray>::from_shape_data(5, 5, pix_in);
        let expected = Image::<bool, Gray>::from_shape_data(5, 5, pix_out);
        let actual = input.dilate(kern.view());
        println!("{:#?}", actual);
        assert_eq!(actual, expected);
        input.dilate_inplace(kern.view());
        assert_eq!(input, expected);
    }
}
