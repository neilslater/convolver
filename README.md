# Convolver

[![Build Status](https://travis-ci.org/neilslater/convolver.png?branch=master)](http://travis-ci.org/neilslater/convolver)

Adds an "inner" convolve operation to NArray floats. It is around 250 times faster than equivalents
in pure Ruby.

Note that convolves based on FFTW3 will be faster still for large arrays with large kernels,
and low ranks (1D, 2D and 3D). At some future point I hope to make this code support convolution
via FFTW3, and use it automatically when it would be the fastest option.

## Installation

Add this line to your application's Gemfile:

    gem 'convolver'

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install convolver

## Usage

Basic convolution:

    a = NArray[0.3,0.4,0.5]
    b = NArray[1.3, -0.5]
    c = Convolver.convolve( b )
    => NArray.float(2): [ 0.19, 0.27 ]

 * Convolver only works on single-precision floats internally. It will cast NArray types to this, if
possible, prior to calculating.
 * The convolution is an "inner" one. The output is smaller than the input, each dimension is reduced
by 1 less than the width of the kernel in the same dimension.
 * Convolver expects input a and kernel b to have the same rank, and for the kernel to be same size
or smaller in all dimensions as the input.

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
