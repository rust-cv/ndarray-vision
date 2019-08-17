# Change Log

## Develop Branch (Unreleased)

### Added
* Padding strategies (`NoPadding`, `ConstantPadding`, `ZeroPadding`)

### Changed
* Integrated Padding strategies into convolutions

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
