use crate::core::{ColourModel, Image};
use crate::format::{Decoder, Encoder};
use num_traits::{Num, NumAssignOps};
use num_traits::cast::{NumCast, FromPrimitive};
use std::fmt::Display;
use std::io::{Error, ErrorKind};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum EncodingType {
    Binary,
    Plaintext,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PpmEncoder {
    encoding: EncodingType,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct PpmDecoder;

impl PpmEncoder {
    /// Create a new PPM encoder or decoder
    pub fn new() -> Self {
        PpmEncoder {
            encoding: EncodingType::Binary,
        }
    }

    /// Creates a new PPM format to encode plain-text. This results in very large
    /// file sizes so isn't recommended in general use
    pub fn plaintext_format() -> Self {
        PpmEncoder {
            encoding: EncodingType::Plaintext,
        }
    }
    
    /// Gets the maximum pixel value in the image across all channels. This is
    /// used in the PPM header
    fn get_max_value<T>(image: &Image<T>) -> Option<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd,
    {
        image
            .data
            .iter()
            .fold(T::zero(), |ref acc, x| if x > acc { *x } else { *acc })
            .to_u8()
    }

    ///! Generate the header string for the image
    fn generate_header(&self, rows: usize, cols: usize, max_value: u8) -> String {
        use EncodingType::*;
        match self.encoding {
            Plaintext => format!("P3\n{} {} {}\n", rows, cols, max_value),
            Binary => format!("P6\n{} {} {}\n", rows, cols, max_value),
        }
    }

    /// Encode the image into the binary PPM format (P6) returning the bytes
    fn encode_binary<T>(&self, image: &Image<T>) -> Vec<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display,
    {
        let image = match image.colour_model() {
            ColourModel::RGB => image,
            _ => panic!("Colour conversions aren't yet supported"),
        };
        let max_val = Self::get_max_value(image).unwrap_or_else(|| 255);

        let mut result = self
            .generate_header(image.rows(), image.cols(), max_val)
            .into_bytes();
        // Not very accurate as a reserve, doesn't factor in max storage for
        // a pixel or spaces. But somewhere between best and worst case
        result.reserve(image.rows() * image.cols() * 5);

        // There is a 70 character line length in PPM using another string to keep track
        for data in image.data.iter() {
            let value = data.to_u8().unwrap_or_else(|| 0);
            result.push(value);
        }
        result
    }

    /// Encode the image into the plaintext PPM format (P3) returning the text as
    /// an array of bytes
    fn encode_plaintext<T>(&self, image: &Image<T>) -> Vec<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display,
    {
        let image = match image.colour_model() {
            ColourModel::RGB => image,
            _ => panic!("Colour conversions aren't yet supported"),
        };
        let max_val = Self::get_max_value(image).unwrap_or_else(|| 255);

        let mut result = self.generate_header(image.rows(), image.cols(), max_val);
        // Not very accurate as a reserve, doesn't factor in max storage for
        // a pixel or spaces. But somewhere between best and worst case
        result.reserve(image.rows() * image.cols() * 5);

        // There is a 70 character line length in PPM using another string to keep track
        let mut temp = String::new();
        let max_margin = 70 - 12;
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


impl<T> Encoder<T> for PpmEncoder
where
    T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display,
{
    fn encode(&self, image: &Image<T>) -> Vec<u8> {
        use EncodingType::*;
        match self.encoding {
            Plaintext => self.encode_plaintext(image),
            Binary => self.encode_binary(image),
        }
    }
}

impl PpmDecoder {
    fn decode_header(bytes: &[u8]) -> (usize, usize) {
        // We don't need the max value for decoding bytes!
        unimplemented!()
    }

    fn decode_binary<T>(bytes: &[u8]) -> std::io::Result<Image<T>>
    where
        T: Copy + Clone + FromPrimitive + Num + NumAssignOps + NumCast + PartialOrd + Display,
    {
        unimplemented!()
    }

    fn decode_plaintext<T>(bytes: &[u8]) -> std::io::Result<Image<T>>
    where
        T: Copy + Clone + FromPrimitive + Num + NumAssignOps + NumCast + PartialOrd + Display,
    {
        let err = || Error::new(ErrorKind::InvalidData, "Error in file encoding");
        // plaintext is easier than binary because the whole thing is a string
        let data = String::from_utf8(bytes.to_vec())
            .map_err(|_| err())?;
        
        let mut rows: Option<usize> = None;
        let mut cols: Option<usize> = None;
        let mut max_val: Option<usize> = None;
        let mut image_bytes = Vec::<T>::new();
        for line in data.lines().filter(|l| !l.starts_with("#")) {

            for value in line.split_whitespace().take_while(|x| !x.starts_with("#")){
                let temp = value.parse::<usize>().map_err(|_| err())?;
                if rows.is_none() {
                    rows = Some(temp);
                } else if cols.is_none() {
                    cols = Some(temp);
                    image_bytes.reserve(rows.unwrap()*cols.unwrap()*3);
                } else if max_val.is_none() {
                    max_val = Some(temp);
                } else {
                    image_bytes.push(T::from_usize(temp)
                                     .unwrap_or_else(|| T::zero()));
                }
            }
        }

        if image_bytes.is_empty() {
            Err(err())
        } else {
            let image = Image::<T>::from_shape_data(rows.unwrap(), 
                                                    cols.unwrap(), 
                                                    ColourModel::RGB,
                                                    image_bytes);
            Ok(image)
        }
    }
}

impl<T> Decoder<T> for PpmDecoder
where
    T: Copy + Clone + FromPrimitive + Num + NumAssignOps + NumCast + PartialOrd + Display,
{
    fn decode(&self, bytes: &[u8]) -> std::io::Result<Image<T>> {
        if bytes.len() < 9 {
            Err(Error::new(
                ErrorKind::InvalidData,
                "File is below minimum size of ppm"
            ))
        } else {
            if bytes.starts_with(b"P3") {
                Self::decode_plaintext(&bytes[2..])
            } else if bytes.starts_with(b"P6") {
                Self::decode_binary(&bytes[2..])
            } else {
                Err(Error::new(
                    ErrorKind::InvalidData,
                    "File is below minimum size of ppm",
                ))
            }
        }
    }
}
