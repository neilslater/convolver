# Convolver

[![CI](https://github.com/neilslater/convolver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/neilslater/convolver/actions/workflows/ci.yml)
[![Gem Version](https://badge.fury.io/rb/convolver.svg)](https://badge.fury.io/rb/convolver)

Convolver calculates cross-correlations between multidimensional
[`Numo::NArray`](https://github.com/yoshoku/numo-narray-alt) values, with
configurable output extents and signal boundary extensions. It chooses between
a direct native implementation for smaller inputs and a
[`Numo::Pocketfft`](https://github.com/yoshoku/numo-pocketfft)-based
implementation for larger inputs.

Version 1.0 replaces the unmaintained `narray` and `fftw3` gems with
`numo-narray-alt` and `numo-pocketfft`.

## Installation

Add Convolver to your application's Gemfile:

```ruby
gem 'convolver'
```

Then run:

```sh
bundle install
```

Alternatively, install it directly:

```sh
gem install convolver
```

No external FFT library is required; PocketFFT is bundled by its Ruby gem.

## Usage

```ruby
require 'convolver'

signal = Numo::SFloat[0.3, 0.4, 0.5]
kernel = Numo::SFloat[1.3, -0.5]

Convolver.convolve(signal, kernel)
# => Numo::SFloat#shape=[2]
#    [0.19, 0.27]
```

With no keywords, Convolver preserves its original valid-correlation behavior:
the signal and kernel must have the same rank, the kernel must be no larger
than the signal in any dimension, and only positions with complete overlap are
returned. The result size in each dimension is:

```ruby
signal_size - kernel_size + 1
```

Inputs are converted to single-precision floats internally, and results are
returned as `Numo::SFloat`. Supplying `Numo::SFloat` inputs avoids conversion in
the direct implementation.

`Convolver.convolve` normally chooses the implementation automatically. The
implementations can also be called directly for application-specific
benchmarking:

```ruby
Convolver.convolve_basic(signal, kernel)
Convolver.convolve_fft(signal, kernel)
```

Both implementations and the automatic method accept the same options:

```ruby
Convolver.convolve(signal, kernel,
                   mode: :same,
                   boundary: :reflect,
                   origin: 0)
```

### Output modes

`mode:` controls the returned extent independently in each dimension:

| Mode | Meaning | Result size |
| --- | --- | --- |
| `:valid` | Kernel overlaps the stored signal completely | `S - K + 1` |
| `:same` | One result aligned with each stored signal position | `S` |
| `:full` | Every kernel position with any stored-signal overlap | `S + K - 1` |

`:valid` is the default and accepts only the default `boundary: :constant`,
`fill_value: 0.0`, and `origin: 0`. `:full` supports constant extension only.
`:same` supports every boundary described below. Kernels larger than the signal
are supported by `:same` and `:full`, but not by `:valid`. `mode:` and
`boundary:` accept only the symbols listed here.

### Boundary extension

For a one-dimensional signal `a b c d`, `boundary:` selects values outside the
stored signal:

| Boundary | Extended sequence |
| --- | --- |
| `:constant` | `k k k k | a b c d | k k k k` |
| `:nearest` | `a a a a | a b c d | d d d d` |
| `:reflect` | `d c b a | a b c d | d c b a` |
| `:mirror` | `d c b | a b c d | c b a` |
| `:wrap` | `a b c d | a b c d | a b c d` |

`fill_value:` sets `k` for `:constant` and defaults to zero. It must not be
passed with another boundary. `:reflect` repeats the edge sample; `:mirror`
does not. All boundary modes work across every dimension, including
length-one axes and extensions wider than the stored signal.

### Kernel origin

`origin:` shifts the kernel anchor for `:same`. It accepts one integer applied
to every dimension or an array with one integer per dimension. For a kernel
dimension of length `K`:

```ruby
anchor = (K / 2) + origin
```

The anchor must remain within the kernel. Positive origins sample farther
toward lower signal indices. With the default origin, an even kernel gives the
extra boundary sample to the lower-index, or left, side: a length-four kernel
uses two samples before and one after the aligned signal position. Nonzero
origins are supported only for `:same`; `:valid` and `:full` require zero.

The estimator methods accept and validate the same options:

```ruby
Convolver.predict_convolve_basic_time(signal, kernel, mode: :same, boundary: :nearest)
Convolver.predict_convolve_fft_time(signal, kernel, mode: :same, boundary: :wrap)
```

`Convolver.convolve_fftw3` remains as a deprecated alias for `convolve_fft` to
ease migration from Convolver 0.x and forwards all options.

## Contributing

Install the development dependencies, then run the main suite:

```sh
bundle install
bundle exec rake
bundle exec rubocop
bundle exec rake c:lint
```

The Ruby specs exercise both the Ruby API and native extension. Additional
native-code checks are available:

```sh
bundle exec rake c:coverage  # Requires GCC and gcovr
bundle exec rake c:sanitize  # Requires Linux and GCC
```

`c:coverage` writes HTML and Cobertura reports under `coverage/c`. CI uploads
the reports as a `c-coverage` artifact. The sanitizer task uses AddressSanitizer
and UndefinedBehaviorSanitizer.

## Contributors

- [Dima Ermilov](https://github.com/adworse) contributed the original Windows
  compilation support.
