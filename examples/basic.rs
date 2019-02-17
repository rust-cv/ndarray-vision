use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::*;
use ndarray_vision::format::*;

use ndarray::{arr1, arr3, Array3};

fn main() {
    let decoder = PpmDecoder::default();
    let image: Image<u8> = decoder.decode_file("test.ppm").unwrap();
    let ppm = PpmEncoder::new();
    let res = ppm.encode_file(&image, "test2.ppm");

    let decoder2 = PpmDecoder::default();
    let image: Image<u8> = decoder.decode_file("test2.ppm").unwrap();
    let ppm = PpmEncoder::new_plaintext_encoder();
    let res = ppm.encode_file(&image, "test3.ppm");
    println!("Save result {:?}", res);
    // Print pixel and image data to show change
    println!("{:?}", image.data);
}
