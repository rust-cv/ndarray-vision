/// Implementation of a Canny Edge Detector and associated types
pub mod canny;
/// Image convolutions in 2D
pub mod conv;
/// Not convolution based image filters
pub mod filter;
/// Common convolution kernels and traits to aid in the building of kernels
pub mod kernels;
/// Sobel operator for edge detection
pub mod sobel;
/// Thresholding functions
pub mod threshold;

pub use canny::*;
pub use conv::*;
pub use filter::*;
pub use kernels::*;
pub use sobel::*;
pub use threshold::*;

/// Common error type for image processing algorithms
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum Error {
    /// Indicates that an error was caused by an image having an unexpected number
    /// of channels. This could be caused by something such as an RGB image being
    /// input to an algorithm that only works on greyscale images
    ChannelDimensionMismatch,
    /// Invalid dimensions to an algorithm - this includes rows and columns and
    /// relationships between the two
    InvalidDimensions,
    /// An invalid parameter has been supplied to an algorithm.
    InvalidParameter,
    /// Numeric error such as an invalid conversion or issues in floating point
    /// math. As `ndarray` and `ndarray-vision` rely on `num_traits` for a lot
    /// of generic functionality this may indicate things such as failed typecasts
    NumericError,
}
