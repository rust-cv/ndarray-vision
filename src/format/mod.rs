use crate::core::Image;
use std::io::prelude::*;
use std::fs::File;

/// Trait for an image encoder
pub trait Encoder<T> {

    pub fn encode(image: &Image<T>) -> Vec<u8>;

    pub fn encode_file(image: &Image<T>, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        file.write_all(encode(image))?;
        Ok(())
    }
}

/// Trait for an image decoder, use this to get an image from a byte stream
pub trait Decoder<T> {
    pub fn decode(bytes: &[u8]) -> std::io::Result<Image<T>>;

    pub fn decode(filename: &str) -> std::io::Result<Image<T>> {
        let bytes = fs::read(filename)?;
        decode(bytes)
    }
}


pub mod netpbm;
