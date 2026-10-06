# Convolver

[![CI](https://github.com/neilslater/convolver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/neilslater/convolver/actions/workflows/ci.yml)
[![Gem Version](https://badge.fury.io/rb/convolver.svg)](https://badge.fury.io/rb/convolver)

Convolver combines numeric sequences or grids with a kernel of weights. Use it
for operations such as smoothing readings, applying an image filter or matching
a pattern. Inputs are [`Numo::NArray`](https://github.com/yoshoku/numo-narray-alt)
arrays. Convolver chooses between direct calculation and
[`PocketFFT`](https://github.com/yoshoku/numo-pocketfft) transforms using estimates
of their cost.

## Installation

Add Convolver to your application's Gemfile:

```ruby
gem 'convolver'
```

Then run `bundle install`, or install it directly with `gem install convolver`.
Ruby 3.3 or newer and a toolchain able to build native extensions are required.
No external FFT library is needed; PocketFFT is bundled by its Ruby gem.

## Start with a small example

A kernel assigns a weight to each value in a moving window. Convolution reverses
those weights within the window; cross-correlation keeps their stored order.
Pass the original kernel to either method.

```ruby
require 'convolver'

signal = Numo::DFloat[1, 2, 4, 8, 16]
kernel = Numo::DFloat[1, 2, 3]

Convolver.convolve(signal, kernel).to_a  # => [11.0, 22.0, 44.0]
Convolver.correlate(signal, kernel).to_a # => [17.0, 34.0, 68.0]
```

The first convolution result is `1*3 + 2*2 + 4*1 = 11`. The corresponding
correlation result is `1*1 + 2*2 + 4*3 = 17`. Neither method modifies its inputs.

Use `convolve` for mathematical convolution, such as applying a smoothing kernel
or combining independent probability distributions. Use `correlate` to compare
windows with a pattern. Correlation here is a sum of products, not a normalized
correlation coefficient.

## Keep the input size or include the edges

By default, Convolver returns only complete windows. `mode:` changes the output
size along each dimension, where `S` is the signal size and `K` the kernel size:

| Mode | Positions included | Result size |
| --- | --- | --- |
| `:valid` (default) | Complete windows inside the signal | `S - K + 1` |
| `:same` | One output for each signal position | `S` |
| `:full` | All positions with any signal overlap | `S + K - 1` |

For example, a centered moving average can keep the input size and reflect
values at the edges:

```ruby
signal = Numo::DFloat[1, 2, 4, 8, 16]
kernel = Numo::DFloat.ones(3) / 3
smoothed = Convolver.convolve(signal, kernel, mode: :same, boundary: :reflect)
smoothed.shape # => [5]
smoothed.to_a.map { |value| value.round(3) } # => [1.333, 2.333, 4.667, 9.333, 13.333]
```

Boundary choices are constant fill, nearest value, reflection with or without
repeating the endpoint, and periodic wrap. `:same` supports all of them;
`:full` supports constant fill only. For nonconstant boundaries, omit
`fill_value:` entirely, even when its value would be zero.

The [calculation rules](docs/rules.md) show each boundary, allowed option
combinations, and how `origin:` aligns odd- and even-length kernels.

## Choose precision and calculation path

The result is `Numo::SFloat` or `Numo::DFloat`, chosen from both inputs.
`dtype: nil` means automatic selection; either floating class can be requested
explicitly:

```ruby
signal = Numo::DFloat[1, 2, 4]
kernel = Numo::DFloat[1, 2]
Convolver.convolve(signal, kernel, dtype: Numo::SFloat).class # => Numo::SFloat
```

Integer inputs are accepted according to the supported-class table in the
rules, but results and arithmetic are floating-point. Convert to a floating
array before dividing to make kernel weights. A later `dtype:` override cannot
recover information already lost to integer division or rounding.

Start with `convolve` or `correlate` for automatic selection. To request a
particular path, use `convolve_basic` / `correlate_basic` for direct calculation
or `convolve_fft` / `correlate_fft` for FFT calculation. Both return the same
shape and class; rounding can differ, so compare values with suitable tolerances.
Neither automatic selection nor the `predict_*_time` estimates guarantee the
fastest method or a measured running time.

## Reference documentation

- [Input and calculation rules](docs/rules.md): exact supported classes,
  precision, output formulas, boundaries, alignment, estimates and memory limits.
- [Terminology](docs/terminology.md): explanations of the words used in the API.

These reference files are included in the gem. From a development checkout,
`bundle exec rake docs:build` builds the API reference and these pages together
in `doc/index.html`.

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
builds the linked pages in `doc/`, checks local link destinations and anchors,
and rejects YARD warnings and undocumented public API objects. Private
implementation classes are marked `@private`. Additional native-code checks are available:

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
