use crate::core::colour_models::*;
use crate::core::traits::PixelBound;
use ndarray::prelude::*;
use ndarray::{s, Data, DataMut, OwnedRepr, ViewRepr};
use num_traits::cast::{FromPrimitive, NumCast};
use num_traits::Num;
use std::marker::PhantomData;

pub type Image<T, C> = ImageBase<OwnedRepr<T>, C>;
pub type ImageView<'a, T, C> = ImageBase<ViewRepr<&'a T>, C>;

/// Basic structure containing an image.
pub struct ImageBase<T, C>
where
    C: ColourModel,
    T: Data,
{
    /// Images are always going to be 3D to handle rows, columns and colour
    /// channels
    ///
    /// This should allow for max compatibility with maths ops in ndarray.
    /// Caution should be taken if performing any operations that change the
    /// number of channels in an image as this may cause other functionality to
    /// perform incorrectly. Use conversions to one of the `Generic` colour models
    /// instead.
    pub data: ArrayBase<T, Ix3>,
    /// Representation of how colour is encoded in the image
    pub(crate) model: PhantomData<C>,
}

impl<T, U, C> ImageBase<U, C>
where
    U: Data<Elem = T>,
    T: Copy + Clone + FromPrimitive + Num + NumCast + PixelBound,
    C: ColourModel,
{
    /// Converts image into a different type - doesn't scale to new pixel bounds
    pub fn into_type<T2>(self) -> Image<T2, C>
    where
        T2: Copy + Clone + FromPrimitive + Num + NumCast + PixelBound,
    {
        let rescale = |x: &T| {
            let scaled = normalise_pixel_value(*x)
                * (T2::max_pixel() - T2::min_pixel())
                    .to_f64()
                    .unwrap_or_else(|| 0.0f64);
            T2::from_f64(scaled).unwrap_or_else(T2::zero) + T2::min_pixel()
        };
        let data = self.data.map(rescale);
        Image::<_, C>::from_data(data)
    }
}

impl<S, T, C> ImageBase<S, C>
where
    S: Data<Elem = T>,
    T: Clone,
    C: ColourModel,
{
    pub fn to_owned(&self) -> Image<T, C> {
        Image {
            data: self.data.to_owned(),
            model: PhantomData,
        }
    }
}

impl<T, C> Image<T, C>
where
    T: Clone + Num,
    C: ColourModel,
{
    /// Construct a new image filled with zeros using the given dimensions and
    /// a colour model
    pub fn new(rows: usize, columns: usize) -> Self {
        Image {
            data: Array3::zeros((rows, columns, C::channels())),
            model: PhantomData,
        }
    }

    /// Given the shape of the image and a data vector create an image. If
    /// the data sizes don't match a zero filled image will be returned instead
    /// of panicking
    pub fn from_shape_data(rows: usize, cols: usize, data: Vec<T>) -> Self {
        let data = Array3::from_shape_vec((rows, cols, C::channels()), data)
            .unwrap_or_else(|_| Array3::<T>::zeros((rows, cols, C::channels())));

        Image {
            data,
            model: PhantomData,
        }
    }
}

impl<T, C> ImageBase<T, C>
where
    T: Data,
    C: ColourModel,
{
    /// Create an image given an existing ndarray
    pub fn from_array(data: ArrayBase<T, Ix3>) -> Self {
        Self {
            data,
            model: PhantomData,
        }
    }
}

impl<T, U, C> ImageBase<T, C>
where
    T: Data<Elem = U>,
    C: ColourModel,
{
    /// Construct the image from a given Array3
    pub fn from_data(data: ArrayBase<T, Ix3>) -> Self {
        Self {
            data,
            model: PhantomData,
        }
    }
    /// Returns the number of rows in an image
    pub fn rows(&self) -> usize {
        self.data.shape()[0]
    }
    /// Returns the number of channels in an image
    pub fn cols(&self) -> usize {
        self.data.shape()[1]
    }

    /// Convenience method to get number of channels
    pub fn channels(&self) -> usize {
        C::channels()
    }

    /// Get a view of all colour channels at a pixels location
    pub fn pixel(&self, row: usize, col: usize) -> ArrayView<U, Ix1> {
        self.data.slice(s![row, col, ..])
    }

    pub fn into_type_raw<C2>(self) -> ImageBase<T, C2>
    where
        C2: ColourModel,
    {
        assert_eq!(C2::channels(), C::channels());
        ImageBase::<T, C2>::from_data(self.data)
    }
}

impl<T, U, C> ImageBase<T, C>
where
    T: DataMut<Elem = U>,
    C: ColourModel,
{
    /// Get a mutable view of a pixels colour channels given a location
    pub fn pixel_mut(&mut self, row: usize, col: usize) -> ArrayViewMut<U, Ix1> {
        self.data.slice_mut(s![row, col, ..])
    }
}

/// Returns a normalised pixel value or 0 if it can't convert the types.
/// This should never fail if your types are good.
pub fn normalise_pixel_value<T>(t: T) -> f64
where
    T: PixelBound + Num + NumCast,
{
    let numerator = (t + T::min_pixel()).to_f64();
    let denominator = (T::max_pixel() - T::min_pixel()).to_f64();

    let numerator = numerator.unwrap_or_else(|| 0.0f64);
    let denominator = denominator.unwrap_or_else(|| 1.0f64);

    numerator / denominator
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn image_consistency_checks() {
        let i = Image::<u8, RGB>::new(1, 2);
        assert_eq!(i.rows(), 1);
        assert_eq!(i.cols(), 2);
        assert_eq!(i.channels(), 3);
        assert_eq!(i.channels(), i.data.shape()[2]);
    }

    #[test]
    fn image_type_conversion() {
        let mut i = Image::<u8, RGB>::new(1, 1);
        i.pixel_mut(0, 0)
            .assign(&arr1(&[u8::max_value(), 0, u8::max_value() / 3]));
        let t: Image<u16, RGB> = i.into_type();
        assert_eq!(
            t.pixel(0, 0),
            arr1(&[u16::max_value(), 0, u16::max_value() / 3])
        );
    }
}
