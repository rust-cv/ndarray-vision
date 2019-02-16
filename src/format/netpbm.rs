use crate::core::{Image, PixelFormat};
use crate::format::{Decoder, Encoder};
use num_traits::{Num, NumAssignOps};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
struct PpmFormat {
    is_plaintext: bool,
}

impl PpmFormat {
    pub fn new() -> Self {
        PpmFormat {
            is_plaintext: false,
        }
    }

    fn get_header_values<T>(image: &Image<T>) -> (usize, usize, u8)
    where
        T: Clone + Num + NumAssignOps,
    {
        unimplemented!()
    }

    fn encode_binary<T>(&self, image: &Image<T>) -> Vec<u8>
    where
        T: Clone + Num + NumAssignOps,
    {
        let (rows, cols, max_val) = Self::get_header_values(image);
        unimplemented!()
    }

    fn encode_plaintext<T>(&self, image: &Image<T>) -> Vec<u8>
    where
        T: Clone + Num + NumAssignOps,
    {
        let (rows, cols, max_val) = Self::get_header_values(image);
        let image = match image.pixel_format() {
            PixelFormat::RGB => image,
            other @ _ => image,
        };
        unimplemented!()
    }
}

impl<T> Encoder<T> for PpmFormat
where
    T: Clone + Num + NumAssignOps,
{
    fn encode(&self, image: &Image<T>) -> Vec<u8> {
        match self.is_plaintext {
            true => self.encode_plaintext(image),
            false => self.encode_binary(image),
        }
    }
}

impl<T> Decoder<T> for PpmFormat
where
    T: Clone + Num + NumAssignOps,
{
    fn decode(&self, bytes: &[u8]) -> std::io::Result<Image<T>> {
        unimplemented!()
    }
}
