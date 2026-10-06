# Convolver terminology

This glossary explains the words used in the [calculation rules](rules.md) and
API reference. See [README](../README.md) for installation and usage.

| Term | Meaning in Convolver |
| --- | --- |
| Signal | The input data being processed: for example, a sequence of readings or a grid of pixel values. It is the first argument. |
| Kernel | The second input array. Its values supply the weights multiplied with a window of signal values. |
| Window | The signal positions paired with the kernel to calculate one output value. A window can include values supplied by a boundary rule. |
| Convolution | Sliding multiplication and addition with the kernel reversed within each window. Call `convolve`; do not reverse the supplied kernel yourself. |
| Cross-correlation | Sliding multiplication and addition with the kernel in its stored order. Call `correlate`. It can measure agreement with a pattern; this is not a normalized correlation coefficient. |
| Dimension, axis, rank | A sequence has one dimension; a grid has two. An axis is one of those dimensions, and rank is their count. Rank here is not matrix rank from linear algebra. |
| Shape | The size along each dimension. Shape `[3, 5]` describes a grid with three rows and five columns, containing 15 values. |
| Scalar | A single value held in a zero-dimensional Numo array. Its shape is `[]`, unlike a one-element sequence with shape `[1]`. |
| Output extent | The size of the returned array along each dimension. `mode:` selects valid, same or full output. |
| Boundary, extension | A rule supplying values beyond the stored signal's edges. Extension does not modify the input array. |
| Fill value | The value supplied outside the signal for `boundary: :constant`. It defaults to zero and must be omitted for other boundaries. |
| Anchor, origin | The anchor is the kernel index used for same-mode alignment. The origin shifts that index from `floor(kernel_size / 2)` along each dimension. |
| Dtype | The class describing how array values are stored. Convolver returns `Numo::SFloat` (single precision) or `Numo::DFloat` (double precision). |
| Promotion | Automatic selection of a floating type from both inputs. DFloat wins if either input maps to it; otherwise SFloat is used. |
| Precision, rounding | Floating types retain a limited number of significant digits. Rounding replaces a value with a nearby representable value; double precision retains more digits than single precision. |
| Tolerance | An allowed numerical difference when comparing results. It accounts for rounding; suitable values depend on the scale of the data and the precision used. |
| View | An array referring to selected or rearranged values of another array, such as a slice or reversed sequence. Convolver uses the view's logical order. |
| Direct calculation | Computing the sliding products and sums directly. Convolver implements this path in its native extension. |
| FFT | Fast Fourier transform: a way to express data in frequency components. The FFT path uses transforms to calculate the same convolution or correlation, with potentially different rounding and memory use. |
| Working buffer | Temporary storage used during calculation. Its size and precision can differ from the returned array. |
| Folding | Adding kernel values that land on the same periodic position. This is preparation used for wrap boundaries, not a change to the stored input. |
| Relative cost, heuristic | An approximate comparison used to choose a calculation path. It is not a measured duration or a guarantee of which path will be fastest. |
| Native limit | A size or arithmetic limit of the underlying native implementation. Passing these checks does not guarantee that physical memory is available. |
