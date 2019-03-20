pub mod conv;
pub mod filter;
pub mod kernels;
pub mod sobel;
pub mod canny;

pub use conv::*;
pub use filter::*;
pub use kernels::*;
pub use sobel::*;
pub use canny::*;


#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub enum Error {
    ChannelDimensionMismatch,
    InvalidDimensions,
    InvalidParameter,
    NumericError,
}
