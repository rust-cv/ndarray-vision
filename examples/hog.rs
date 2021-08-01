use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::*;
use ndarray_vision::format::*;
use ndarray_vision::features::*;
use std::env::current_exe;
use std::path::PathBuf;

fn hog_features(img: &Image<f64, Gray>) -> Image<u8, Gray> {
    let hog = HogExtractor::create()
        .build();

    let features = hog.get_features(img);
    hog.visualise_features((img.rows(), img.cols()), features.view())
}

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

        let hog = hog_features(&image);

        let mut lena = PathBuf::from(&root);
        lena.push("images/lena-hog.ppm");
        let ppm = PpmEncoder::new_plaintext_encoder();
        ppm.encode_file(&hog.into(), lena)
            .expect("Unable to encode ppm");
    }
}

