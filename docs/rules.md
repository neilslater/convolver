# Convolver input and calculation rules

These rules describe Convolver 4. They apply to both convolution and
cross-correlation. See the [terminology](terminology.md) for unfamiliar words,
or [README](../README.md) for installation and a quick example.

## Choose an operation

Use `Convolver.convolve` for mathematical convolution and `Convolver.correlate`
for cross-correlation. Both slide a kernel over a signal, multiply paired values
and add the products. Convolution reverses the kernel within each window;
correlation keeps its stored order. Pass the kernel in its original order in
either case: Convolver performs the named operation itself.

```ruby
require 'convolver'

signal = Numo::DFloat[1, 2, 4, 8, 16]
kernel = Numo::DFloat[1, 2, 3]

Convolver.convolve(signal, kernel).to_a  # => [11.0, 22.0, 44.0]
Convolver.correlate(signal, kernel).to_a # => [17.0, 34.0, 68.0]
```

For the first window, convolution calculates `1*3 + 2*2 + 4*1 = 11`.
Correlation calculates `1*1 + 2*2 + 4*3 = 17`.

Each operation has three entry points:

| Choice | Convolution | Cross-correlation |
| --- | --- | --- |
| Automatic choice | `convolve` | `correlate` |
| Direct calculation | `convolve_basic` | `correlate_basic` |
| FFT calculation | `convolve_fft` | `correlate_fft` |

Start with the automatic methods. They estimate which implementation is likely
to be faster; they do not guarantee the fastest choice for every machine or
input. Explicit methods are useful for measurements or when a workload needs a
particular calculation path.

## Supply compatible arrays

Both inputs must be nonempty Numo arrays with the same number of dimensions,
up to `Convolver::MAX_RANK` (16). Their sizes may differ. Two zero-dimensional
arrays are supported and their scalar values are multiplied.

The supported input classes and their automatic floating types are:

| Input class in `Numo` | Automatic floating type |
| --- | --- |
| `SFloat`, `Int8`, `UInt8`, `Int16`, `UInt16` | `Numo::SFloat` |
| `DFloat`, `Int32`, `UInt32`, `Int64`, `UInt64` | `Numo::DFloat` |

Inputs must be instances of these concrete classes. Ordinary Ruby arrays,
`Numo::Bit`, complex arrays, `Numo::RObject` and custom subclasses are rejected,
even when `dtype:` is supplied. Supported array views, including slices and
reversed views, are accepted. Neither input is modified.

## Choose precision

By default, Convolver maps each input using the table above. If either maps to
`Numo::DFloat`, the result is DFloat; otherwise it is SFloat. Swapping the input
types does not change this choice. `dtype: nil` means the same as omitting it.

Use `dtype: Numo::SFloat` or `dtype: Numo::DFloat` to choose explicitly. These
classes and `nil` are the only accepted dtype values. Both inputs and any
constant fill are converted to the chosen type before preparing boundary
values or calculating. The fill value does not influence the choice of dtype.

Choose floating inputs before dividing to make kernel weights. Integer division
can discard fractions before Convolver sees them:

```ruby
Numo::Int16.ones(3) / 3  # => integer weights [0, 0, 0]
Numo::DFloat.ones(3) / 3 # => floating weights, each approximately one third
```

A later dtype override cannot recover information already lost to integer
division or floating-point rounding. Integer inputs produce approximate floating
calculations, not integer results. DFloat can represent every 32-bit integer,
but not every 64-bit integer. Neither result type guarantees exact integer
convolution.

The direct path multiplies and adds in the chosen precision. The FFT path uses
double-precision working buffers, then returns the chosen result type. Shapes
and result classes agree across paths, but rounding can differ. Compare numeric
results with a tolerance appropriate to the data and precision.

Normal floating conversion applies, including overflow to infinity and
underflow. NaN and infinity are accepted, but their propagation can differ
between calculation paths.

## Choose output size

`mode:` controls the size of the result. In this table, `S` is the signal size
and `K` the kernel size along one dimension. Apply the rule to each dimension.

| Mode | Positions included | Result size |
| --- | --- | --- |
| `:valid` (default) | Only windows fully inside the stored signal | `S - K + 1` |
| `:same` | One output aligned with each stored signal position | `S` |
| `:full` | All positions with any stored-signal overlap | `S + K - 1` |

In valid mode, the kernel must fit inside the signal in every dimension. Larger
kernels are permitted in same and full modes. Equal numbers of dimensions are
still required; a one-dimensional kernel is not automatically applied to each
row of a two-dimensional signal.

For one-dimensional valid results, returned index `p` starts at zero:

```text
convolve(signal, kernel)[p]  = sum_j signal[p + K - 1 - j] * kernel[j]
correlate(signal, kernel)[p] = sum_j signal[p + j] * kernel[j]
```

Here `j` runs from zero to `K - 1`. For multiple dimensions, apply the index
rule along every dimension and sum over all kernel positions. Convolver accepts
real inputs only, so correlation has no visible complex-conjugation step.

## Choose values beyond the edges

`boundary:` supplies values outside the stored signal. For a signal `a b c d`,
the following table shows examples immediately before and after it:

| Boundary | Before | Stored signal | After |
| --- | --- | --- | --- |
| `:constant` (default) | `k k k k` | `a b c d` | `k k k k` |
| `:nearest` | `a a a a` | `a b c d` | `d d d d` |
| `:reflect` | `d c b a` | `a b c d` | `d c b a` |
| `:mirror` | `d c b` | `a b c d` | `c b a` |
| `:wrap` | `a b c d` | `a b c d` | `a b c d` |

Reflect repeats the edge sample; mirror does not. Wrap treats the signal as
repeating. These rules work along every dimension, including dimensions of
length one and extensions wider than the stored signal.

For constant boundaries, `fill_value:` sets `k` and must be a real numeric
value. Omitting it uses zero. For all other boundaries, omit `fill_value:`
entirely: even an explicit `fill_value: 0` is rejected.

Some generated method signatures show `UNSPECIFIED_FILL`. This is Convolver's
internal marker for an omitted argument, not a value callers should pass. It
lets Convolver distinguish omission from an explicitly supplied zero.

The permitted combinations are:

| Mode | Boundary | Fill | Origin |
| --- | --- | --- | --- |
| `:valid` | `:constant` only | Omitted or zero | Zero |
| `:same` | Any listed boundary | Real numeric for constant; omit for others | Within the range below |
| `:full` | `:constant` only | Omitted or real numeric | Zero |

Mode and boundary names must be the symbols shown, not strings.

## Align the kernel

`origin:` adjusts kernel alignment in same mode. Supply an integer for all
dimensions, or an array containing one integer per dimension. The default is
zero. For a kernel dimension of length `K`:

```text
anchor = floor(K / 2) + origin
```

The anchor must be a kernel index: `0 <= anchor < K`. For example, a kernel of
length three allows origins -1, 0 and 1; length four allows -2, -1, 0 and 1.
Valid and full modes require zero origins.

For a same-mode output at position `i`, with boundary values supplied as above:

```text
convolution[i] = sum_j extended_signal[i + anchor - j] * kernel[j]
correlation[i] = sum_j extended_signal[i + j - anchor] * kernel[j]
```

With origin zero, odd-length kernels use the same alignment for both operations.
For even-length kernels, correlation needs the extra boundary sample before the
signal, and convolution needs it after. Increasing the origin moves convolution
windows toward higher signal indices and correlation windows toward lower ones.

## Interpret estimates and failures

`predict_convolve_basic_time`, `predict_convolve_fft_time`,
`predict_correlate_basic_time` and `predict_correlate_fft_time` accept the same
inputs and options as the calculation methods. They return comparative cost
estimates, not measured elapsed times. Smaller estimates suggest less work.
They do not execute the convolution or allocate its result array.

The estimates account for input conversion, boundary preparation and periodic
kernel folding. Ranks one through three use calibrated profiles for arm64 macOS
and x86_64/aarch64 Linux. Other platforms and higher ranks use a general
heuristic. These profiles describe broad platform families, not measurements
of the caller's particular CPU. Automatic selection can change as estimates
improve.

Invalid inputs or option combinations raise `ArgumentError`. Calculations and
estimators reject unrepresentable planned dimensions or buffers with
`RangeError`. FFT checks include conservative native integer and working-buffer
limits. Automatic calculation can use direct execution if FFT exceeds its
limits and the direct path is still valid.

These checks do not reserve RAM or guarantee enough memory is available.
Ordinary allocation failures still raise `NoMemoryError`.

For older applications, see the [README](../README.md) sections on the version 3
precision migration and version 2 operation-name migration.
