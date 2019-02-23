use ndarray::{Array3, Ix3};
use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::*;
use ndarray_vision::format::*;
use ndarray_vision::processing::*;
use std::env::current_exe;
use std::path::PathBuf;

fn main() {
    if let Ok(mut root) = current_exe() {
        root.pop();
        root.pop();
        root.pop();
        root.pop();
        let mut lena = PathBuf::from(&root);
        lena.push("images/lena.ppm");

        let decoder = PpmDecoder::default();
        let image: Image<u8, _> = decoder.decode_file(lena).expect("Couldn't open Lena.ppm");

        let boxkern: Array3<f64> =
            BoxLinearFilter::build(Ix3(3, 3, 3)).expect("Was unable to construct filter");

        let mut image: Image<f64, _> = image.into_type();

        let _ = image
            .conv2d_inplace(boxkern.view())
            .expect("Poorly sized kernel");
        // There's no u8: From<f64> so I've done this to hack things
        image.data *= 255.0f64;

        let mut lena = PathBuf::from(&root);
        lena.push("images/lenablur.ppm");

        let ppm = PpmEncoder::new_plaintext_encoder();
        ppm.encode_file(&image, lena).expect("Unable to encode ppm");
    }
}
