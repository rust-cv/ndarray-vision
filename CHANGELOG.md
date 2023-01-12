# Change Log

## [Unreleased]

### Added

- The `ThresholdApplyExt` trait to apply user-defined threshold
- The `threshold_apply` method to the `ArrayBase` and `Image` types

## [0.4.0] 2022-02-17

### Changed

- Remove discrete levels - this overflowed with the 64 and 128 bit types

## [0.3.0] 2021-11-24

### Changed

- Fixed orientation of sobel filters
- Fixed remove limit on magnitude in sobel magnitude calculation

## [0.2.0] 2020-06-06

### Added

- Padding strategies (`NoPadding`, `ConstantPadding`, `ZeroPadding`)
- Threshold module with Otsu and Mean threshold algorithms
- Image transformations and functions to create affine transform matrices
- Type alias `Image` for `ImageBase<OwnedRepr<T>, _>` replicated old `Image` type
- Type alias `ImageView` for `ImageBase<ViewRepr<&'a T>, _>`
- Morphology module with dilation, erosion, union and intersection of binary images

### Changed

- Integrated Padding strategies into convolutions
- Updated `ndarray-stats` to 0.2.0 adding `noisy_float` for median change
- [INTERNAL] Disabled code coverage due to issues with tarpaulin and native libraries
- Renamed `Image` to `ImageBase` which can take any implementor of the ndaray `Data` trait
- Made images have `NoPadding` by default
- No pad behaviour now keeps pixels near the edges the same as source value instead of making them black
- Various performance enhancements in convolution and canny functions

## [0.1.1] - 2019-07-31

### Changed

- Applied zero padding by default in convolutions

## [0.1.0] - 2019-03-24

### Added

- Image type
- Colour Models (RGB, Gray, HSV, CIEXYZ, Channel-less)
- Histogram equalisation
- Image convolutions
- `PixelBound` type to aid in rescaling images
- Canny edge detector
- `KernelBuilder` and `FixedDimensionKernelBuilder` to create kernels
- Builder implementations for Sobel, Gaussian, Box Linear filter, Laplace
- Median filter
- Sobel Operator
- PPM encoding and decoding for images
