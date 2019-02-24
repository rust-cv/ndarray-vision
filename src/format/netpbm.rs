use crate::core::{rescale_pixel_value, Image, PixelBound, RGB};
use crate::format::{Decoder, Encoder};
use num_traits::cast::{FromPrimitive, NumCast};
use num_traits::{Num, NumAssignOps};
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

impl Default for PpmEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Implements the encoder trait for the PpmEncoder.
///
/// The ColourModel type argument is locked to RGB - this prevents calling
/// RGB::into::<RGB>() unnecessarily which is unavoidable until trait specialisation is
/// stabilised.
impl<T> Encoder<T, RGB> for PpmEncoder
where
    T: Copy
        + Clone
        + Num
        + NumAssignOps
        + NumCast
        + PartialOrd
        + Display
        + PixelBound
        + FromPrimitive,
{
    fn encode(&self, image: &Image<T, RGB>) -> Vec<u8> {
        use EncodingType::*;
        match self.encoding {
            Plaintext => self.encode_plaintext(image),
            Binary => self.encode_binary(image),
        }
    }
}

impl PpmEncoder {
    /// Create a new PPM encoder or decoder
    pub fn new() -> Self {
        PpmEncoder {
            encoding: EncodingType::Binary,
        }
    }

    /// Creates a new PPM format to encode plain-text. This results in very large
    /// file sizes so isn't recommended in general use
    pub fn new_plaintext_encoder() -> Self {
        PpmEncoder {
            encoding: EncodingType::Plaintext,
        }
    }

    /// Gets the maximum pixel value in the image across all channels. This is
    /// used in the PPM header
    fn get_max_value<T>(image: &Image<T, RGB>) -> Option<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display + PixelBound,
    {
        image
            .data
            .iter()
            .fold(T::zero(), |ref acc, x| if x > acc { *x } else { *acc })
            .to_u8()
    }

    ///! Generate the header string for the image
    fn generate_header(self, rows: usize, cols: usize, max_value: u8) -> String {
        use EncodingType::*;
        match self.encoding {
            Plaintext => format!("P3\n{} {} {}\n", rows, cols, max_value),
            Binary => format!("P6\n{} {} {}\n", rows, cols, max_value),
        }
    }

    /// Encode the image into the binary PPM format (P6) returning the bytes
    fn encode_binary<T>(self, image: &Image<T, RGB>) -> Vec<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display + PixelBound,
    {
        let max_val = Self::get_max_value(image).unwrap_or_else(|| 255);

        let mut result = self
            .generate_header(image.rows(), image.cols(), max_val)
            .into_bytes();

        result.reserve(result.len() + (image.rows() * image.cols() * 3));

        for data in image.data.iter() {
            let value = (rescale_pixel_value(*data) * 255.0f64) as u8;
            result.push(value);
        }
        result
    }

    /// Encode the image into the plaintext PPM format (P3) returning the text as
    /// an array of bytes
    fn encode_plaintext<T>(self, image: &Image<T, RGB>) -> Vec<u8>
    where
        T: Copy + Clone + Num + NumAssignOps + NumCast + PartialOrd + Display + PixelBound,
    {
        let max_val = 255;

        let mut result = self.generate_header(image.rows(), image.cols(), max_val);
        // Not very accurate as a reserve, doesn't factor in max storage for
        // a pixel or spaces. But somewhere between best and worst case
        result.reserve(image.rows() * image.cols() * 5);

        // There is a 70 character line length in PPM using another string to keep track
        let mut temp = String::new();
        let max_margin = 70 - 12;
        temp.reserve(max_margin);

        for data in image.data.iter() {
            let value = (rescale_pixel_value(*data) * 255.0f64) as u8;
            temp.push_str(&format!("{} ", value));
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

/// Implements the decoder trait for the PpmDecoder.
///
/// The ColourModel type argument is locked to RGB - this prevents calling
/// RGB::into::<RGB>() unnecessarily which is unavoidable until trait specialisation is
/// stabilised.
impl<T> Decoder<T, RGB> for PpmDecoder
where
    T: Copy
        + Clone
        + Num
        + NumAssignOps
        + NumCast
        + PartialOrd
        + Display
        + PixelBound
        + FromPrimitive,
{
    fn decode(&self, bytes: &[u8]) -> std::io::Result<Image<T, RGB>> {
        if bytes.len() < 9 {
            Err(Error::new(
                ErrorKind::InvalidData,
                "File is below minimum size of ppm",
            ))
        } else if bytes.starts_with(b"P3") {
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

impl PpmDecoder {
    /// Decodes a PPM header getting (rows, cols, maximum value) or returning
    /// an io::Error if the header is malformed
    fn decode_header(bytes: &[u8]) -> std::io::Result<(usize, usize, usize)> {
        let err = || Error::new(ErrorKind::InvalidData, "Error in file header");
        let mut keep = true;
        let bytes = bytes
            .iter()
            .filter(|x| {
                if *x == &b'#' {
                    keep = false;
                    false
                } else if !keep {
                    if *x == &b'\n' || *x == &b'\r' {
                        keep = true;
                    }
                    false
                } else {
                    true
                }
            })
            .map(|x| *x)
            .collect::<Vec<_>>();

        if let Ok(s) = String::from_utf8(bytes) {
            let res = s
                .split_whitespace()
                .map(|x| x.parse::<usize>().unwrap_or(0))
                .collect::<Vec<_>>();
            if res.len() == 3 {
                Ok((res[0], res[1], res[2]))
            } else {
                Err(err())
            }
        } else {
            Err(err())
        }
    }

    fn decode_binary<T>(bytes: &[u8]) -> std::io::Result<Image<T, RGB>>
    where
        T: Copy
            + Clone
            + Num
            + NumAssignOps
            + NumCast
            + PartialOrd
            + Display
            + PixelBound
            + FromPrimitive,
    {
        let err = || Error::new(ErrorKind::InvalidData, "Error in file encoding");
        const WHITESPACE: &[u8] = b" \t\n\r";

        let mut image_bytes = Vec::<T>::new();

        let mut last_saw_whitespace = false;
        let mut is_comment = false;
        let mut val_count = 0;
        let header_end = bytes
            .iter()
            .position(|&b| {
                if b == b'#' {
                    is_comment = true;
                } else if is_comment {
                    if b == b'\r' || b == b'\n' {
                        is_comment = false;
                    }
                } else if last_saw_whitespace && !WHITESPACE.contains(&b) {
                    val_count += 1;
                    last_saw_whitespace = false;
                } else if WHITESPACE.contains(&b) {
                    last_saw_whitespace = true;
                }
                val_count == 3 && WHITESPACE.contains(&b)
            })
            .ok_or_else(err)?;

        let (rows, cols, max_val) = Self::decode_header(&bytes[0..header_end])?;
        for b in bytes.iter().skip(header_end + 1) {
            let real_pixel = (*b as f64) * (255.0f64 / (max_val as f64));
            image_bytes.push(T::from_u8(real_pixel as u8).unwrap_or_else(T::zero));
        }

        if image_bytes.is_empty() || image_bytes.len() != (rows * cols * 3) {
            Err(err())
        } else {
            let image = Image::<T, RGB>::from_shape_data(rows, cols, image_bytes);
            Ok(image)
        }
    }

    fn decode_plaintext<T>(bytes: &[u8]) -> std::io::Result<Image<T, RGB>>
    where
        T: Copy
            + Clone
            + Num
            + NumAssignOps
            + NumCast
            + PartialOrd
            + Display
            + PixelBound
            + FromPrimitive,
    {
        let err = || Error::new(ErrorKind::InvalidData, "Error in file encoding");
        // plaintext is easier than binary because the whole thing is a string
        let data = String::from_utf8(bytes.to_vec()).map_err(|_| err())?;

        let mut rows = -1;
        let mut cols = -1;
        let mut max_val = -1;
        let mut image_bytes = Vec::<T>::new();
        for line in data.lines().filter(|l| !l.starts_with('#')) {
            for value in line.split_whitespace().take_while(|x| !x.starts_with('#')) {
                let temp = value.parse::<isize>().map_err(|_| err())?;
                if rows < 0 {
                    rows = temp;
                } else if cols < 0 {
                    cols = temp;
                    image_bytes.reserve((rows * cols * 3) as usize);
                } else if max_val < 0 {
                    max_val = temp;
                } else {
                    let real_pixel = (temp as f64) * (255.0f64 / (max_val as f64));
                    image_bytes.push(T::from_f64(real_pixel).unwrap_or_else(T::zero));
                }
            }
        }
        if image_bytes.is_empty() || image_bytes.len() != ((rows * cols * 3) as usize) {
            Err(err())
        } else {
            let image = Image::<T, RGB>::from_shape_data(rows as usize, cols as usize, image_bytes);
            Ok(image)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::colour_models::*;
    use ndarray::arr1;

    #[test]
    fn max_value_test() {
        let full_range = "P3 1 1 255 0 255 0";
        let clamped = "P3 1 1 1 0 1 0";

        let decoder = PpmDecoder::default();
        let full_image: Image<u8, RGB> = decoder.decode(full_range.as_bytes()).unwrap();
        let clamp_image: Image<u8, RGB> = decoder.decode(clamped.as_bytes()).unwrap();

        assert_eq!(full_image, clamp_image);
        assert_eq!(full_image.pixel(0, 0), arr1(&[0, 255, 0]));
    }

    #[test]
    fn encoding_consistency() {
        let image_str = "P3 
            3 3 255 
            255 255 255  0 0 0  255 0 0 
            0 255 0  0 0 255  255 255 0
            0 255 255  127 127 127  0 0 0";

        let decoder = PpmDecoder::default();
        let image: Image<u8, RGB> = decoder.decode(image_str.as_bytes()).unwrap();

        let encoder = PpmEncoder::new();
        let image_bytes = encoder.encode(&image);

        let restored: Image<u8, RGB> = decoder.decode(&image_bytes).unwrap();

        assert_eq!(image, restored);

        let encoder = PpmEncoder::new_plaintext_encoder();
        let image_bytes = encoder.encode(&image);
        let restored: Image<u8, RGB> = decoder.decode(&image_bytes).unwrap();

        assert_eq!(image, restored);
    }

    #[test]
    fn binary_comments() {
        let image_str = "P3 
            3 3 255 
            255 255 255  0 0 0  255 0 0 
            0 255 0  0 0 255  255 255 0
            0 255 255  127 127 127  0 0 0";

        let decoder = PpmDecoder::default();
        let image: Image<u8, RGB> = decoder.decode(image_str.as_bytes()).unwrap();

        let encoder = PpmEncoder::new();
        let mut image_bytes = encoder.encode(&image);
        let comment = b"# This is a comment\n";
        for i in 0..comment.len() {
            image_bytes.insert(2 + i, comment[i]);
        }
        let restored: Image<u8, RGB> = decoder.decode(&image_bytes).unwrap();

        assert_eq!(image, restored);
    }
}
