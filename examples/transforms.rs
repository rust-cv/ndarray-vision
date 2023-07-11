use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::*;
use ndarray_vision::format::*;
use ndarray_vision::transform::affine::*;
use ndarray_vision::transform::*;
use std::env::current_exe;
use std::f64::consts::FRAC_PI_4;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

fn get_cameraman() -> Option<Image<u8, RGB>> {
    if let Ok(mut root) = current_exe() {
        root.pop();
        root.pop();
        root.pop();
        root.pop();
        let mut cameraman = PathBuf::from(&root);
        cameraman.push("images/cameraman.ppm");

        let decoder = PpmDecoder::default();
        let image: Image<u8, _> = decoder
            .decode_file(cameraman)
            .expect("Couldn't open cameraman.ppm");
        Some(image)
    } else {
        None
    }
}

fn main() {
    let cameraman = get_cameraman().expect("Couldn't load cameraman");

    // Create transformation matrix
    let x = 0.5 * (cameraman.cols() as f64) - 0.5;
    let y = 0.5 * (cameraman.rows() as f64) - 0.5;
    let trans =
        transform_from_2dmatrix(rotate_around_centre(FRAC_PI_4, (x, y)).dot(&scale(0.7, 0.7)));

    let transformed = cameraman.transform(&trans, None).expect("Transform failed");

    // save
    let path = Path::new("transformed_cameraman.png");
    let file = File::create(path).expect("Couldn't create output file");
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, transformed.cols() as u32, transformed.rows() as u32);
    encoder.set_color(png::ColorType::RGB);
    encoder.set_depth(png::BitDepth::Eight);

    println!(
        "Writing image with resolution {}x{}",
        transformed.cols(),
        transformed.rows()
    );

    let mut writer = encoder.write_header().expect("Failed to write file header");
    if let Some(data) = transformed.data.view().to_slice() {
        writer
            .write_image_data(data)
            .expect("Failed to write image data");
    } else {
        println!("Failed to get image slice");
    }
}
