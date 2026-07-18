# Convolver

[![Gem Version](https://badge.fury.io/rb/convolver.png)](http://badge.fury.io/rb/convolver)
[![Code Climate](https://codeclimate.com/github/neilslater/convolver.png)](https://codeclimate.com/github/neilslater/convolver)

Calculates discrete convolution between two multi-dimensional arrays of floats.
See http://en.wikipedia.org/wiki/Convolution

## Installation

### Dependency: FFTW3

Before you install *convolver*, you should install the FFTW3 library on your system.
See http://www.fftw.org/ for details.

On macOS with Homebrew, the `fftw3` Ruby gem needs to be told where Homebrew
installed the headers and libraries:

    brew install fftw
    bundle config set --local build.fftw3 "--with-fftw3-dir=$(brew --prefix fftw)"
    bundle install

The local Bundler setting is written to `.bundle/config`, which is intentionally
not committed because the Homebrew prefix depends on the machine.

### Known warning with Ruby 3.4

The first use of `NArray` may emit this warning:

    warning: undefining the allocator of T_DATA class NArray

This comes from the legacy native allocation API used by `narray` 0.6.1.2. Ruby
disables the inherited allocator when the first native NArray object is created.
The warning does not affect convolver's results; removing it properly requires a
fix in `narray` or a future migration to a maintained NArray implementation.

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
possible, prior to calculating. For best speed, use NArray.sfloat arrays.

 * The output is smaller than the input, it only contains fully-calculated values. The output size
is the original size, minus the kernel size, plus 1, in each dimension.

 * Convolver expects input a and kernel b to have the same rank, and for the kernel to be same size
or smaller in all dimensions as the input.

 * Convolver.convolve will try to choose the faster of two approaches it has coded. In general,
small convolutions are processed directly by multiplying out all combinations and summing them,
and large convolutions are processed using FFTW3 to convert to frequency space where convolution
is simpler and faster to calculate, then convert back.

## Contributing

The Ruby specs also exercise the native extension. Native-code quality checks are available as
separate Rake tasks:

    bundle exec rake c:lint
    bundle exec rake c:coverage  # GCC and gcovr required
    bundle exec rake c:sanitize

`c:coverage` writes an HTML report to `coverage/c/index.html` and Cobertura XML to
`coverage/c/cobertura.xml`. The pull-request workflow uploads these reports as a `c-coverage`
artifact and includes the text summary in the job summary. `c:sanitize` rebuilds the extension with
AddressSanitizer and UBSan and currently requires Linux and GCC.

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

## Contributors

 * [Dima Ermilov](https://github.com/adworse) contributed fix to support compiling under Windows.
