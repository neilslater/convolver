# Convolver

[![Build Status](https://travis-ci.org/neilslater/convolver.png?branch=master)](http://travis-ci.org/neilslater/convolver)

Adds a convolve operation to NArray floats. It is around 250 times faster than equivalents
in pure Ruby.

The gem makes convolution via FFTW3 library available. This is faster for convolutions with
larger kernels and signals. The relationship is complex, but as a rule of thumb, the kernel
needs to be around 1000 entries or larger before it is worth switching to FFTW3-based convolves.

## Planned features

The *convolver* gem will eventually contain a basic kit for creating, training and running convolutional
neural networks. As a side effect of this plan, it will also contain efficient code for
calculating signal convolutions for other types of analysis.

## Installation

### Dependency: FFTW3

Before you install *convolver*, you should install FFTW3. See http://www.fftw.org/ for details.

### Installing the gem

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
    c = Convolver.convolve( a, b )
    => NArray.float(2): [ 0.19, 0.27 ]

 * Convolver only works on single-precision floats internally. It will cast NArray types to this, if
possible, prior to calculating.

 * The output is smaller than the input, each dimension is reduced by 1 less than the width of the
kernel in the same dimension.

 * Convolver expects input a and kernel b to have the same rank, and for the kernel to be same size
or smaller in all dimensions as the input.

FFTW3 convolution:

    a = NArray[0.3,0.4,0.5]
    b = NArray[1.3, -0.5]
    c = Convolver.convolve_fftw3( a, b )
    => NArray.float(2): [ 0.19, 0.27 ]

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request
