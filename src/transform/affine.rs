use super::Transform;
use ndarray::{array, prelude::*};
use ndarray_linalg::Inverse;

/// converts a matrix into an equivalent `AffineTransform`
pub fn transform_from_2dmatrix(in_array: Array2<f64>) -> AffineTransform {
    let transform = match in_array.inv() {
        Ok(inv) => AffineTransform {
            matrix2d_transform: in_array.clone(),
            matrix2d_transform_inverse: inv,
            inverse_exists: true,
        },
        Err(e) => AffineTransform {
            matrix2d_transform: in_array.clone(),
            matrix2d_transform_inverse: Array2::zeros((2, 2)),
            inverse_exists: false,
        },
    };
    return transform;
}

/// a linear transform of an image represented by either size 2x2
/// or 3x3 ( right column is a translation and projection ) matrix applied to the image index
/// coordinates
pub struct AffineTransform {
    matrix2d_transform: Array2<f64>,
    matrix2d_transform_inverse: Array2<f64>,
    inverse_exists: bool,
}

fn source_coordinate(p: (f64, f64), trans: ArrayView2<f64>) -> (f64, f64) {
    let p = match trans.shape()[0] {
        2 => array![[p.0], [p.1]],
        3 => array![[p.0], [p.1], [1.0]],
        _ => unreachable!(),
    };

    let result = trans.dot(&p);
    let x = result[[0, 0]];
    let y = result[[1, 0]];
    let w = match trans.shape()[0] {
        2 => 1.0,
        3 => result[[2, 0]],
        _ => unreachable!(),
    };
    if (w - 1.0).abs() > std::f64::EPSILON {
        (x / w, y / w)
    } else {
        (x, y)
    }
}

impl Transform for AffineTransform {
    fn apply(&self, p: (f64, f64)) -> (f64, f64) {
        return source_coordinate(p, self.matrix2d_transform.view());
    }

    fn apply_inverse(&self, p: (f64, f64)) -> (f64, f64) {
        return source_coordinate(p, self.matrix2d_transform_inverse.view());
    }

    fn inverse_exists(&self) -> bool {
        return self.inverse_exists;
    }
}

/// describes the Axes to use in rotation_3d
/// X and Y correspond to the image index coordinates and
/// Z is perpendicular out of the image plane
pub enum Axes {
    X,
    Y,
    Z,
}

/// generates a 2d matrix describing a rotation around a 2d coordinate
pub fn rotate_around_centre(radians: f64, centre: (f64, f64)) -> Array2<f64> {
    translation(centre.0, centre.1)
        .dot(&rotation_3d(radians, Axes::Z))
        .dot(&translation(-centre.0, -centre.1))
}

/// generates a matrix describing 2d rotation around origin
pub fn rotation_2d(radians: f64) -> Array2<f64> {
    let s = radians.sin();
    let c = radians.cos();
    array![[c, -s], [s, c]]
}

/// generates a 3x3 matrix describing a rotation around either the index coordinate axes
/// (X,Y) or in the perpendicular axes to the image (Z)
pub fn rotation_3d(radians: f64, ax: Axes) -> Array2<f64> {
    let s = radians.sin();
    let c = radians.cos();

    match ax {
        Axes::X => array![[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        Axes::Y => array![[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        Axes::Z => array![[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// generates a matrix describing translation in the image index space
pub fn translation(x: f64, y: f64) -> Array2<f64> {
    array![[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]]
}

/// generates a matrix describing scaling in image index space
pub fn scale(x: f64, y: f64) -> Array2<f64> {
    array![[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, 1.0]]
}

/// generates a matrix describing shear in image index space
pub fn shear(x: f64, y: f64) -> Array2<f64> {
    array![[1.0, x, 0.0], [y, 1.0, 0.0], [0.0, 0.0, 1.0]]
}
