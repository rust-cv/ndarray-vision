# ndarray-vision

[![Build Status](https://travis-ci.org/xd009642/ndarray-vision.svg?branch=master)](https://travis-ci.org/xd009642/ndarray-vision)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/xd009642/ndarray-vision/badge.svg?branch=master)](https://coveralls.io/github/xd009642/ndarray-vision?branch=master)

This project is a computer vision library built on top of ndarray. This project
is a work in progress. Basic image encoding/decoding and processing are
currently implemented.

See the examples and tests for basic usage.

# Features

* Conversions between Grayscale, RGB, HSV and CIEXYZ
* Image convolutions and common kernels (box linear, gaussian, laplace)
* Median filtering
* Sobel operator
* Canny Edge Detection
* Histogram Equalisation
* Encoding and decoding PPM (binary or plaintext)
