# Change Log

## Develop Branch (Unreleased)

### Added
* Padding strategies (`NoPadding`, `ConstantPadding`, `ZeroPadding`)
* Threshold module with Otsu and Mean threshold algorithms
* Image transformations and functions to create affine transform matrices

### Changed
* Integrated Padding strategies into convolutions
* Updated `ndarray-stats` to 0.2.0 adding `noisy_float` for median change
* [INTERNAL] Disabled code coverage due to issues with tarpaulin and native libraries

### Removed 

## 0.1.1

### Added

### Changed
* Applied zero padding by default in convolutions

### Removed 

## 0.1.0 (First release)

### Added
* Image type
* Colour Models (RGB, Gray, HSV, CIEXYZ, Channel-less)
* Histogram equalisation
* Image convolutions
* `PixelBound` type to aid in rescaling images
* Canny edge detector
* `KernelBuilder` and `FixedDimensionKernelBuilder` to create kernels
* Builder implementations for Sobel, Gaussian, Box Linear filter, Laplace
* Median filter
* Sobel Operator
* PPM encoding and decoding for images
