
use ndarray_vision::core::*;

use ndarray::{arr1, arr3, Array3};

fn main() {
    let format = PixelFormat::RGB;
    
    // Create an empty image
    let mut image = Image::<f64>::new(5, 10, format);
    
    // Pixels all assigned to zero initially
    println!("{:?}", image.pixel(0, 0));


    // Set top left pixel to RGB 1.0, 0.0, 0.0
    image.pixel_mut(0,0).assign(&arr1(&[1.0, 0.0, 0.0]));

    let k: Array3::<f64> = arr3(&[[[0.,0.,0.],[4.,4.,4.],[0.,0.,0.]],
                        [[0.,0.,0.],[10.,10.,10.],[0.,0.,0.]],
                        [[0.,0.,0.],[4.,4.,4.],[0.,0.,0.]]]);
    image.conv(k.view());
    // Print pixel and image data to show change
    println!("{:?}", image.pixel(0, 0));
    println!("{:?}", image.data);
}
