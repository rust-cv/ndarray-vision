use ndarray_vision::core::*;
use ndarray_vision::format::netpbm::*;
use ndarray_vision::format::*;
use ndarray_vision::processing::*;
use std::env::current_exe;
use std::path::PathBuf;

fn canny_edges(img: &Image<f64, Gray>) -> Image<f64, Gray> {
    let x = CannyBuilder::<f64>::new()
        .lower_threshold(0.3)
        .upper_threshold(0.5)
        .blur((5, 5), [0.4, 0.4])
        .build();
    let res = img.canny_edge_detector(x).expect("Failed to run canny");

    Image::from_data(res.data.mapv(|x| if x { 1.0 } else { 0.0 }))
}

fn main() {
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

        let image: Image<f64, _> = image.into_type();
        let image: Image<_, Gray> = image.into();

        let canny = canny_edges(&image);

        let image = image.apply_sobel().expect("Error in sobel");
        // back to RGB
        let image: Image<_, RGB> = image.into();
        let mut cameraman = PathBuf::from(&root);
        cameraman.push("images/cameraman-sobel.ppm");

        let ppm = PpmEncoder::new_plaintext_encoder();
        ppm.encode_file(&image, cameraman)
            .expect("Unable to encode ppm");

        let mut cameraman = PathBuf::from(&root);
        cameraman.push("images/cameraman-canny.ppm");
        ppm.encode_file(&canny.into(), cameraman)
            .expect("Unable to encode ppm");
    }
}
