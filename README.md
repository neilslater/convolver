# Convolver

[![CI](https://github.com/neilslater/convolver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/neilslater/convolver/actions/workflows/ci.yml)
[![Gem Version](https://badge.fury.io/rb/convolver.svg)](https://badge.fury.io/rb/convolver)

Convolver calculates mathematical convolution and cross-correlation between
multidimensional
[`Numo::NArray`](https://github.com/yoshoku/numo-narray-alt) values. Both
operations support configurable output extents and signal boundary extensions.
Convolver chooses between a direct native implementation for smaller inputs
and a [`Numo::Pocketfft`](https://github.com/yoshoku/numo-pocketfft)-based
implementation for larger inputs.

## Installation

Add Convolver to your application's Gemfile:

```ruby
gem 'convolver'
```

Then run `bundle install`, or install the gem directly with
`gem install convolver`. No external FFT library is required; PocketFFT is
bundled by its Ruby gem.

## Usage

`convolve` calculates mathematical discrete convolution. `correlate`
calculates cross-correlation, following the public naming convention used by
NumPy and SciPy:

```ruby
require 'convolver'

signal = Numo::SFloat[0.3, 0.4, 0.5]
kernel = Numo::SFloat[1.3, -0.5]

Convolver.convolve(signal, kernel)
# => Numo::SFloat#shape=[2]
#    [0.37, 0.45]

Convolver.correlate(signal, kernel)
# => Numo::SFloat#shape=[2]
#    [0.19, 0.27]
```

For real one-dimensional inputs, the two valid operations are:

```text
correlate(signal, kernel)[p] = sum_j signal[p + j] * conjugate(kernel[j])
convolve(signal, kernel)[n]  = sum_j signal[n - j] * kernel[j]
```

The formulas apply component-wise to multidimensional arrays. Correlation
conventionally conjugates the second operand, although Convolver's current
real input contract makes conjugation invisible.

With no keywords, the signal and kernel must have the same rank, the kernel
must be no larger than the signal in any dimension, and only positions with
complete overlap are returned. Both inputs determine the floating result dtype,
or you can select it explicitly with `dtype:`.

The automatic, direct, and FFT implementations are available for both
operations:

```ruby
Convolver.convolve(signal, kernel)
Convolver.convolve_basic(signal, kernel)
Convolver.convolve_fft(signal, kernel)

Convolver.correlate(signal, kernel)
Convolver.correlate_basic(signal, kernel)
Convolver.correlate_fft(signal, kernel)
```

Every calculation method accepts the same options:

```ruby
Convolver.convolve(signal, kernel,
                   mode: :same,
                   boundary: :reflect,
                   origin: 0)
```

### Input and result precision

`dtype:` accepts `Numo::SFloat`, `Numo::DFloat`, or `nil` (automatic, the
default). Automatic selection maps each input type using this table, then
chooses DFloat if either input maps to DFloat, otherwise SFloat:

| Input type | Automatic floating type |
| --- | --- |
| `SFloat`, `Int8`, `UInt8`, `Int16`, `UInt16` | `Numo::SFloat` |
| `DFloat`, `Int32`, `UInt32`, `Int64`, `UInt64` | `Numo::DFloat` |

The input type names above are classes in `Numo`. Both operands must be instances
of these concrete classes. Bit, RObject, complex arrays, custom subclasses and
ordinary Ruby arrays are rejected, even with an explicit `dtype:`. Integer
results and complex arithmetic are not supported.

```ruby
signal = Numo::SFloat[1, 2, 3]
kernel = Numo::DFloat[0.25, 0.75]
Convolver.convolve(signal, kernel)                     # DFloat result
Convolver.convolve(signal, kernel, dtype: Numo::SFloat) # SFloat result
```

Both inputs are cast to the selected dtype before extension, folding or
arithmetic. Inputs are never mutated. `fill_value` is cast to this same dtype
and never promotes the result. Ordinary floating conversion applies, including
rounding, overflow to infinity and underflow; NaN and infinity are not rejected.
Non-finite propagation and rounding can differ between algorithms.

Direct products and accumulation use the selected precision. PocketFFT uses
double-precision working buffers for either result dtype. The returned class
is independent of algorithm selection. Integer inputs become approximate
floating computations: DFloat represents every 32-bit integer, but not every
64-bit integer, and neither dtype guarantees exact integer convolution.

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

### Kernel origin and alignment

`origin:` shifts the stored-kernel anchor for `:same`. It accepts one integer
applied to every dimension or an array with one integer per dimension. For a
kernel dimension of length `K`:

```text
anchor = floor(K / 2) + origin

correlation[i] = sum_j extended_signal[i + j - anchor] * kernel[j]
convolution[i] = sum_j extended_signal[i + anchor - j] * kernel[j]
```

The anchor must remain within the kernel. Correlation pads `anchor` samples
before the signal and `K - 1 - anchor` after it; convolution swaps those
widths. Odd centered kernels therefore align alike. For an even centered
kernel, correlation puts the extra sample before the signal and convolution
puts it after the signal. Positive origins move correlation toward lower signal
indices and convolution toward higher signal indices.

Nonzero origins are supported only for `:same`; `:valid` and `:full` require
zero.

### Estimators

The estimator methods accept and validate the same options as their calculation
families:

```ruby
Convolver.predict_convolve_basic_time(signal, kernel, mode: :same, boundary: :nearest)
Convolver.predict_convolve_fft_time(signal, kernel, mode: :same, boundary: :wrap)

Convolver.predict_correlate_basic_time(signal, kernel, mode: :same, boundary: :nearest)
Convolver.predict_correlate_fft_time(signal, kernel, mode: :same, boundary: :wrap)
```

Estimates are machine-dependent heuristics, including input conversion, boundary
preparation and periodic kernel folding. Automatic selection avoids FFT planning
when a cheap lower bound already favors direct calculation. Ranks one through
three use calibrated profiles on arm64 macOS and x86_64/aarch64 Linux; other
platforms and higher ranks retain the general heuristic. Algorithm choices
can change as estimates improve; benchmark the explicit methods for workloads
where the choice matters.

Calculations and estimators reject unrepresentable buffer sizes with `RangeError`
before allocation. FFT planning also checks conservative PocketFFT integer and
work-buffer limits, including full complex staging for real inverse transforms.
Automatic selection uses direct calculation when FFT exceeds its limits and the
direct path remains valid. These checks do not reserve memory; ordinary allocator
exhaustion still raises `NoMemoryError`.

### Migrating from version 3

Version 4 changes default result types and values: DFloat and 32/64-bit integer
inputs now select DFloat. Pass `dtype: Numo::SFloat` to request single-precision
inputs and results. This retains the version 3 result type, but does not promise
identical numerical results: version 4 consistently casts both inputs before
processing, whereas version 3's direct and FFT paths could round differently.
The input-class restrictions listed above also apply with a dtype override.

### Migrating from version 2

Version 2's `convolve*` methods calculated cross-correlation. Version 3 corrects
the terminology and changes every `convolve*` method to mathematical
convolution. Asymmetric kernels make the result change visible.

To preserve version 2 results, rename the complete method family:

| Version 2 | Version 3 equivalent |
| --- | --- |
| `convolve` | `correlate` |
| `convolve_basic` | `correlate_basic` |
| `convolve_fft` | `correlate_fft` |
| `predict_convolve_basic_time` | `predict_correlate_basic_time` |
| `predict_convolve_fft_time` | `predict_correlate_fft_time` |
| `convolve_fftw3` | `correlate_fft` |

`convolve_fftw3` has been removed. No `cross_correlate` aliases are provided;
the standard `correlate` name denotes cross-correlation explicitly documented
above.

## Contributing

Install the development dependencies, then run the complete local gate:

```sh
bundle install
bundle exec rake
bundle exec rubocop
bundle exec ncs-rubocop-conf-audit
bundle exec rake c:lint
bundle exec bundle-audit check --update
bundle exec rake docs:check
```

The Ruby specs exercise both the Ruby API and native extension and enforce 95%
line and branch coverage for the Ruby library. Dependency auditing refreshes
the advisory database and requires network access. The documentation check
rejects YARD warnings and undocumented public API objects; private implementation
classes are marked `@private`. Additional native-code checks are available:

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
