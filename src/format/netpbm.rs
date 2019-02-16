use crate::core::{Image, ColourModel};
use crate::format::{Decoder, Encoder};
use num_traits::{Num, NumAssignOps, cast::NumCast};
use std::fmt::Display;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct PpmFormat {
    is_plaintext: bool,
}

impl PpmFormat {
    /// Create a new PPM encoder or decoder
    pub fn new() -> Self {
        PpmFormat {
            is_plaintext: false,
        }
    }

    pub fn plaintext_file() -> Self {
        PpmFormat {
            is_plaintext: true
        }
    }

    fn get_max_value<T>(image: &Image<T>) -> Option<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd,
    {
        image.data
            .iter()
            .fold(T::zero(), |ref acc, x| {
                if x > acc {
                    *x
                } else {
                    *acc
                }).to_u8()
    }


    fn encode_binary<T>(&self, image: &Image<T>) -> Vec<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd,
    {
        let max_val = Self::get_max_value(image);
        unimplemented!()
    }


    fn encode_plaintext<T>(&self, image: &Image<T>) -> Vec<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display,
    {
        let mut result = String::from("P3 ");

        let image = match image.colour_model() {
            ColourModel::RGB => image,
            _ => panic!("Colour conversions aren't yet supported"),
        }; 
        let max_val = Self::get_max_value(image).unwrap_or_else(|| 255);

        // Not very accurate as a reserve, doesn't factor in max storage for
        // a pixel or spaces. But somewhere between best and worst case
        result.reserve(image.rows()*image.cols()*5);
        result.push_str(&format!("\n{} {} {}\n", image.rows(), image.cols(), max_val));
        
        // There is a 70 character line length in PPM using another string to keep track 
        let mut temp = String::new();
        let max_margin = 70-12;
        temp.reserve(max_margin);

        for data in image.data.iter() {
            temp.push_str(&format!("{} ", data));
            if temp.len() > max_margin {
                result.push_str(&temp);
                result.push('\n');
                temp.clear();
            }
        }
        if !temp.is_empty() {
            result.push_str(&temp);
        }
        result.into_bytes()
    }
}



impl<T> Encoder<T> for PpmFormat
where
    T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display,
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
    T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display,
{
    fn decode(&self, bytes: &[u8]) -> std::io::Result<Image<T>> {
        unimplemented!()
    }
}
