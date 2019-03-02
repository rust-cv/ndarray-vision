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

        let image: Image<f64, _> = image.into_type();
        let image: Image<_, Gray> = image.into();

        let image = image.apply_sobel().expect("Error in sobel");

        // back to RGB
        let image: Image<_, RGB> = image.into();
        let mut lena = PathBuf::from(&root);
        lena.push("images/lenaedges.ppm");

        let ppm = PpmEncoder::new_plaintext_encoder();
        ppm.encode_file(&image, lena).expect("Unable to encode ppm");
    }
}
