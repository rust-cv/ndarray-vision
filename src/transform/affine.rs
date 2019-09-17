use ndarray::{array, prelude::*};

pub enum Axes {
    X,
    Y,
    Z,
}

pub fn rotation_2d(radians: f64) -> Array2<f64> {
    let s = radians.sin();
    let c = radians.cos();
    array![[c, -s], [s, c]]
}

pub fn rotation_3d(radians: f64, ax: Axes) -> Array2<f64> {
    let s = radians.sin();
    let c = radians.cos();

    match ax {
        Axes::X => array![[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        Axes::Y => array![[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        Axes::Z => array![[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

pub fn translation_2d(x: f64, y: f64) -> Array2<f64> {
    translation(x, y, None)
}

pub fn translation(x: f64, y: f64, z: Option<f64>) -> Array2<f64> {
    if let Some(z) = z {
        array![[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]]
    } else {
        array![[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]]
    }
}
