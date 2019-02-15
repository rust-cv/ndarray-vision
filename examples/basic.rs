
use ndarray_vision::core::*;

use ndarray::{arr1, arr3, Array3};

fn main() {
    let format = PixelFormat::RGB;
    
    // Create an empty image
    let mut image = Image::<f64>::new(6, 6, format);
    
    // Set top left pixel to RGB 1.0, 0.0, 0.0
    image.pixel_mut(0,0).assign(&arr1(&[1.0, 0.0, 0.0]));
    
    println!("{:?}", image.data);

    let mean = 1.0f64/9.0f64;
    let kernel = Array3::from_elem((3,3,3), mean);
    image.conv_inplace(kernel.view());
    
    // Print pixel and image data to show change
    println!("{:?}", image.data);
}
