# Convolver

[![Gem Version](https://badge.fury.io/rb/convolver.svg)](https://badge.fury.io/rb/convolver)

Convolver calculates valid cross-correlations between multidimensional
[`Numo::NArray`](https://github.com/yoshoku/numo-narray-alt) values. It chooses
between a direct native implementation for smaller inputs and a
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

The signal and kernel must have the same rank, and the kernel must be no larger
than the signal in any dimension. Convolver returns only positions at which the
kernel overlaps the signal completely. The result size in each dimension is:

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

`Convolver.convolve_fftw3` remains as a deprecated alias for `convolve_fft` to
ease migration from Convolver 0.x.

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
