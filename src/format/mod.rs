use crate::core::traits::PixelBound;
use crate::core::*;
use num_traits::cast::{FromPrimitive, NumCast};
use num_traits::{Num, NumAssignOps};
use std::fmt::Display;
use std::fs::{read, File};
use std::io::prelude::*;

/// Trait for an image encoder
pub trait Encoder<T, C>
where
    T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display + PixelBound,
    C: ColourModel,
{
    /// Encode an image into a sequence of bytes for the given format
    fn encode(&self, image: &Image<T, C>) -> Vec<u8>;

    /// Encode an image saving it to the file at filename. This function shouldn't
    /// add an extension preferring the user to do that instead.
    fn encode_file(&self, image: &Image<T, C>, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        file.write_all(&self.encode(image))?;
        Ok(())
    }
}

/// Trait for an image decoder, use this to get an image from a byte stream
pub trait Decoder<T, C>
where
    T: Copy
        + Clone
        + FromPrimitive
        + Num
        + NumAssignOps
        + NumCast
        + PartialOrd
        + Display
        + PixelBound,
    C: ColourModel,
{
    /// From the bytes decode an image, will perform any scaling or conversions
    /// required to represent elements with type T.
    fn decode(&self, bytes: &[u8]) -> std::io::Result<Image<T, C>>;
    /// Given a filename decode an image performing any necessary conversions.
    fn decode_file(&self, filename: &str) -> std::io::Result<Image<T, C>> {
        let bytes = read(filename)?;
        self.decode(&bytes)
    }
}

pub mod netpbm;
