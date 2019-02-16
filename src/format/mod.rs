use crate::core::Image;
use num_traits::{Num, NumAssignOps};
use std::fs::{read, File};
use std::io::prelude::*;

/// Trait for an image encoder
pub trait Encoder<T>
where
    T: Clone + Num + NumAssignOps,
{
    /// Encode an image into a sequence of bytes for the given format
    fn encode(&self, image: &Image<T>) -> Vec<u8>;

    /// Encode an image saving it to the file at filename. This function shouldn't
    /// add an extension preferring the user to do that instead.
    fn encode_file(&self, image: &Image<T>, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        file.write_all(&self.encode(image))?;
        Ok(())
    }
}

/// Trait for an image decoder, use this to get an image from a byte stream
pub trait Decoder<T>
where
    T: Clone + Num + NumAssignOps,
{
    /// From the bytes decode an image, will perform any scaling or conversions
    /// required to represent elements with type T.
    fn decode(&self, bytes: &[u8]) -> std::io::Result<Image<T>>;
    /// Given a filename decode an image performing any necessary conversions.
    fn decode_file(&self, filename: &str) -> std::io::Result<Image<T>> {
        let bytes = read(filename)?;
        self.decode(&bytes)
    }
}

pub mod netpbm;
