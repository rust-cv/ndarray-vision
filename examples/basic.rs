use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::PpmFormat;
use ndarray_vision::format::Encoder;

use ndarray::{arr1, arr3, Array3};

fn main() {
    let format = ColourModel::RGB;

    // Create an empty image
    let mut image = Image::<u8>::new(6, 6, format);

    // Set top left pixel to RGB 1.0, 0.0, 0.0
    image.pixel_mut(0, 0).assign(&arr1(&[255, 0, 0]));

    let ppm = PpmFormat::new();
    let res = ppm.encode_file(&image, "test.ppm");
    println!("Save result {:?}", res);
    // Print pixel and image data to show change
    println!("{:?}", image.data);
}
