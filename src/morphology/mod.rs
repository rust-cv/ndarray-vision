use crate::core::*;
use ndarray::{prelude::*, DataMut, Zip};

pub trait MorphologyExt {
    type Output;

    fn erode(&self, kernel: ArrayView2<bool>) -> Self::Output;

    fn erode_inplace(&mut self, kernel: ArrayView2<bool>);

    fn dilate(&self, kernel: ArrayView2<bool>) -> Self::Output;

    fn dilate_inplace(&mut self, kernel: ArrayView2<bool>);

    fn union(&self, other: &Self) -> Self::Output;

    fn union_inplace(&mut self, other: &Self);

    fn intersection(&self, other: &Self) -> Self::Output;

    fn intersection_inplace(&mut self, other: &Self);
}

impl<U> MorphologyExt for ArrayBase<U, Ix3>
where
    U: DataMut<Elem = bool>,
{
    type Output = Array<bool, Ix3>;

    fn erode(&self, kernel: ArrayView2<bool>) -> Self::Output {
        let sh = kernel.shape();
        let (ro, co) = kernel_centre(sh[0], sh[1]);
        let mut result = Self::Output::from_elem(self.dim(), false);
        if self.shape()[0] >= sh[0] && self.shape()[1] >= sh[1] {
            Zip::indexed(self.slice(s![.., .., 0]).windows(kernel.dim())).for_each(
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
            Zip::indexed(self.slice(s![.., .., 0]).windows(kernel.dim())).for_each(
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

    fn intersection(&self, other: &Self) -> Self::Output {
        self & other
    }

    fn intersection_inplace(&mut self, other: &Self) {
        *self &= other;
    }
}

impl<U, C> MorphologyExt for ImageBase<U, C>
where
    U: DataMut<Elem = bool>,
    C: ColourModel,
{
    type Output = Image<bool, C>;

    fn erode(&self, kernel: ArrayView2<bool>) -> Self::Output {
        Self::Output::from_data(self.data.erode(kernel))
    }

    fn erode_inplace(&mut self, kernel: ArrayView2<bool>) {
        self.data.erode_inplace(kernel);
    }

    fn dilate(&self, kernel: ArrayView2<bool>) -> Self::Output {
        Self::Output::from_data(self.data.dilate(kernel))
    }

    fn dilate_inplace(&mut self, kernel: ArrayView2<bool>) {
        self.data.dilate_inplace(kernel);
    }

    fn union(&self, other: &Self) -> Self::Output {
        Self::Output::from_data(self.data.union(&other.data))
    }

    fn union_inplace(&mut self, other: &Self) {
        self.data.union_inplace(&other.data);
    }

    fn intersection(&self, other: &Self) -> Self::Output {
        Self::Output::from_data(self.data.intersection(&other.data))
    }

    fn intersection_inplace(&mut self, other: &Self) {
        self.data.intersection_inplace(&other.data);
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
        assert_eq!(actual, expected);
        input.dilate_inplace(kern.view());
        assert_eq!(input, expected);
    }

    #[test]
    fn simple_erosion() {
        let pix_out = vec![
            false, false, false, false, false, false, false, false, false, false, false, false,
            true, false, false, false, false, false, false, false, false, false, false, false,
            false,
        ];
        let pix_in = vec![
            false, false, false, false, false, false, true, true, true, false, false, true, true,
            true, false, false, true, true, true, false, false, false, false, false, false,
        ];

        let kern = arr2(&[[true, true, true], [true, true, true], [true, true, true]]);

        let mut input = Image::<bool, Gray>::from_shape_data(5, 5, pix_in);
        let expected = Image::<bool, Gray>::from_shape_data(5, 5, pix_out);
        let actual = input.erode(kern.view());
        assert_eq!(actual, expected);
        input.erode_inplace(kern.view());
        assert_eq!(input, expected);
    }

    #[test]
    fn simple_intersect() {
        let a = vec![false, false, false, true, true, true, false, false, false];
        let b = vec![false, true, false, false, true, false, false, true, false];
        let mut a = Image::<bool, Gray>::from_shape_data(3, 3, a);
        let b = Image::<bool, Gray>::from_shape_data(3, 3, b);

        let exp = vec![false, false, false, false, true, false, false, false, false];
        let expected = Image::<bool, Gray>::from_shape_data(3, 3, exp);
        let c = a.intersection(&b);

        assert_eq!(c, expected);
        assert_eq!(a.intersection(&b), b.intersection(&a));

        a.intersection_inplace(&b);
        assert_eq!(a, c);
    }

    #[test]
    fn simple_union() {
        let a = vec![false, false, false, true, true, true, false, false, false];
        let b = vec![false, true, false, false, true, false, false, true, false];
        let mut a = Image::<bool, Gray>::from_shape_data(3, 3, a);
        let b = Image::<bool, Gray>::from_shape_data(3, 3, b);

        let exp = vec![false, true, false, true, true, true, false, true, false];
        let expected = Image::<bool, Gray>::from_shape_data(3, 3, exp);
        let c = a.union(&b);

        assert_eq!(c, expected);
        assert_eq!(a.union(&b), b.union(&a));

        a.union_inplace(&b);
        assert_eq!(a, c);
    }
}
