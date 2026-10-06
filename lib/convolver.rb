# frozen_string_literal: true

require 'numo/narray/alt'
require 'numo/pocketfft'
require 'convolver/convolver'
require 'convolver/version'

# @!macro [new] convolver_overview
#   Combine numeric sequences or grids with a kernel of weights.
#
#   Start with {.convolve} for mathematical convolution or {.correlate} for
#   cross-correlation. Both slide a window over the signal, multiply paired
#   values and add the products. Convolution reverses the kernel within each
#   window; correlation keeps its stored order. Pass the original kernel to
#   either method: Convolver handles the orientation.
#
#   All calculation methods return a new Numo::SFloat or Numo::DFloat array.
#   Inputs are not modified. The automatic methods choose a calculation path;
#   explicit direct and FFT methods are available for comparing performance.
#
#   @example Compare the first complete window
#     signal = Numo::DFloat[1, 2, 4, 8, 16]
#     kernel = Numo::DFloat[1, 2, 3]
#     Convolver.convolve(signal, kernel).to_a  # => [11.0, 22.0, 44.0]
#     Convolver.correlate(signal, kernel).to_a # => [17.0, 34.0, 68.0]
#   @see file:docs/rules.md Complete input and calculation rules
#   @see file:docs/terminology.md Terminology
module Convolver
  # Maximum number of dimensions supported by the implementations.
  MAX_RANK = 16

  require 'convolver/operation_plan'
  require 'convolver/linear_fft_operation'
  require 'convolver/circular_fft_operation'
  require 'convolver/fft_estimator'
  require 'convolver/operation_execution'

  class << self
    # Calculate mathematical convolution, choosing a likely faster calculation path.
    #
    # Each result adds the products from one signal window, with the kernel
    # reversed within that window. By default, only complete windows are returned.
    # Use mode: :same to keep the signal's shape, or mode: :full to include
    # partly overlapping positions at both ends of each dimension.
    #
    # Inputs must be nonempty supported Numo arrays with equal numbers of
    # dimensions, up to {MAX_RANK}. Two zero-dimensional arrays multiply directly.
    # Both input types determine the floating result type unless dtype overrides
    # it. See the rules below for the exact supported classes and combinations.
    #
    # UNSPECIFIED_FILL in the signature means the keyword was omitted. For a
    # constant boundary this uses zero; other boundaries require omission and
    # reject even an explicit fill_value: 0. Do not pass the internal marker.
    #
    # @param signal [Numo::NArray] sequence or grid to process
    # @param kernel [Numo::NArray] weights, supplied in their original order
    # @param mode [:valid, :same, :full] which output positions to return
    # @param boundary [:constant, :nearest, :reflect, :mirror, :wrap]
    #   how to supply values beyond the signal's edges
    # @param fill_value [Numeric] real value outside a constant boundary;
    #   omission uses zero. Valid mode requires zero; other boundaries reject
    #   this keyword even when explicitly zero. The internal default marker
    #   denotes omission and is not a caller-supplied value.
    # @param origin [Integer, Array<Integer>] shift from the kernel midpoint,
    #   shared across dimensions or one integer per dimension; same mode only
    #   permits a nonzero shift, within the kernel's index range
    # @param dtype [Class, nil] Numo::SFloat or Numo::DFloat; nil chooses from
    #   both inputs. SFloat and 8/16-bit integers select SFloat; DFloat and
    #   32/64-bit integers select DFloat. The wider candidate wins.
    # @return [Numo::SFloat, Numo::DFloat] a new mathematical convolution result
    # @raise [ArgumentError] if inputs or options are invalid
    # @raise [RangeError] if no usable calculation path fits native size limits
    # @raise [NoMemoryError] if memory allocation fails despite passing size checks
    # @example Keep the signal size and reflect at its edges
    #   signal = Numo::DFloat[1, 2, 4, 8, 16]
    #   kernel = Numo::DFloat.ones(3) / 3
    #   result = Convolver.convolve(signal, kernel, mode: :same, boundary: :reflect)
    #   result.shape # => [5]
    #   result.to_a.map { |value| value.round(3) } # => [1.333, 2.333, 4.667, 9.333, 13.333]
    # @see file:docs/rules.md Supported arrays, precision, modes and alignment
    def convolve(signal, kernel, mode: :valid, boundary: :constant,
                 fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).automatic
    end

    # Compare a signal with a kernel in its stored order using cross-correlation.
    #
    # Each result is the sum of paired products from one window. This is useful
    # for matching a pattern; it is not a normalized correlation coefficient.
    # Chooses a likely faster path, using the same input/option rules as {.convolve}.
    # Even-length kernels have different same-mode alignment for the two operations;
    # see the reference rules before comparing their output positions.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] a new cross-correlation result
    # @raise (see convolve)
    # @example Compare three complete windows with an asymmetric pattern
    #   signal = Numo::DFloat[1, 2, 4, 8, 16]
    #   kernel = Numo::DFloat[1, 2, 3]
    #   Convolver.correlate(signal, kernel).to_a # => [17.0, 34.0, 68.0]
    # @see file:docs/rules.md Complete input and calculation rules
    def correlate(signal, kernel, mode: :valid, boundary: :constant,
                  fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).automatic
    end

    # Calculate convolution directly from the sliding products and sums.
    #
    # Products and accumulation use the selected result precision. This method
    # always uses the native direct path. For automatic selection use {.convolve}.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] a new mathematical convolution result
    # @raise [ArgumentError] if inputs or options are invalid
    # @raise [RangeError] if direct dimensions or buffers exceed native limits
    # @raise [NoMemoryError] if memory allocation fails despite passing size checks
    # @example Request single-precision input conversion and results
    #   signal = Numo::DFloat[1, 2, 4]
    #   result = Convolver.convolve_basic(signal, Numo::DFloat[1, 2], dtype: Numo::SFloat)
    #   result.to_a # => [4.0, 8.0]
    # @see file:docs/rules.md Shared options and precision rules
    def convolve_basic(signal, kernel, mode: :valid, boundary: :constant,
                       fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic
    end

    # Calculate cross-correlation directly from the sliding products and sums.
    #
    # Keeps the kernel in stored order and uses the selected precision for
    # products and accumulation. For automatic selection use {.correlate}.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] a new cross-correlation result
    # @raise (see convolve_basic)
    # @example Use an asymmetric kernel without reversing it
    #   Convolver.correlate_basic(Numo::DFloat[1, 2, 4], Numo::DFloat[1, 2]).to_a
    #   # => [5.0, 10.0]
    # @see file:docs/rules.md Shared options and alignment rules
    def correlate_basic(signal, kernel, mode: :valid, boundary: :constant,
                        fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic
    end

    # Calculate convolution using PocketFFT transforms.
    #
    # FFT working buffers use double precision for either result dtype. The
    # returned shape and class follow the same rules as {.convolve}; rounding
    # and propagation of non-finite values can differ from the direct path.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] a new mathematical convolution result
    # @raise [ArgumentError] if inputs or options are invalid
    # @raise [RangeError] if FFT dimensions, buffers or arithmetic exceed native limits
    # @raise [NoMemoryError] if memory allocation fails despite passing size checks
    # @example Include every partly overlapping position
    #   result = Convolver.convolve_fft(Numo::DFloat[1, 2, 3], Numo::DFloat[1, 2], mode: :full)
    #   expected = Numo::DFloat[1, 4, 7, 6]
    #   (result - expected).abs.max < 1e-12 # => true
    # @see file:docs/rules.md Shared options, precision and memory limits
    def convolve_fft(signal, kernel, mode: :valid, boundary: :constant,
                     fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft
    end

    # Calculate cross-correlation using PocketFFT transforms.
    #
    # Keeps the kernel in stored order. Working buffers use double precision;
    # the returned dtype is selected from the inputs or the explicit override.
    # Compare results with a tolerance, as for {.convolve_fft}.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] a new cross-correlation result
    # @raise (see convolve_fft)
    # @example Check the result with a floating-point tolerance
    #   result = Convolver.correlate_fft(Numo::DFloat[1, 2, 4], Numo::DFloat[1, 2])
    #   (result - Numo::DFloat[5, 10]).abs.max < 1e-12 # => true
    # @see file:docs/rules.md Shared options, precision and memory limits
    def correlate_fft(signal, kernel, mode: :valid, boundary: :constant,
                      fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft
    end

    # Estimates the relative cost of {.convolve_fft}.
    #
    # Returns a rough comparative cost, not elapsed seconds. Includes input
    # conversion and boundary preparation without running convolution or
    # allocating its result. Profiles describe platform families, not a measured
    # speed for this particular CPU. Compare with {.predict_convolve_basic_time}.
    # @param (see convolve)
    # @return [Float] estimated relative cost on the selected platform profile
    # @raise [ArgumentError] if inputs or options are invalid
    # @raise [RangeError] if the planned FFT path exceeds native limits
    # @example Obtain an estimate without calculating a result array
    #   cost = Convolver.predict_convolve_fft_time(Numo::DFloat.ones(32), Numo::DFloat.ones(3))
    #   cost.finite? && cost >= 0 # => true
    # @see file:docs/rules.md Interpretation and limits of estimates
    def predict_convolve_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                  fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft_time
    end

    # Estimates the relative cost of {.correlate_fft}.
    # Returns a comparative cost without executing correlation or allocating its
    # result. The estimate includes preparation and is not a timing measurement.
    # Compare with {.predict_correlate_basic_time} for the same inputs/options.
    # @param (see convolve)
    # @return [Float] estimated relative cost on the selected platform profile
    # @raise (see predict_convolve_fft_time)
    # @example Estimate periodic correlation work
    #   cost = Convolver.predict_correlate_fft_time(Numo::DFloat.ones(32), Numo::DFloat.ones(3),
    #                                                mode: :same, boundary: :wrap)
    #   cost.finite? && cost >= 0 # => true
    # @see file:docs/rules.md Interpretation and limits of estimates
    def predict_correlate_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                   fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft_time
    end

    # Estimates the relative cost of {.convolve_basic}.
    # Returns a comparative cost without executing convolution or allocating its
    # result. Smaller costs suggest less work, not a guaranteed elapsed time.
    # Compare with {.predict_convolve_fft_time} for the same inputs/options.
    # @param (see convolve)
    # @return [Float] estimated relative cost on the selected platform profile
    # @raise [ArgumentError] if inputs or options are invalid
    # @raise [RangeError] if the planned direct path exceeds native limits
    # @example Estimate direct convolution work
    #   cost = Convolver.predict_convolve_basic_time(Numo::DFloat.ones(32), Numo::DFloat.ones(3))
    #   cost.finite? && cost >= 0 # => true
    # @see file:docs/rules.md Interpretation and limits of estimates
    def predict_convolve_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                    fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic_time
    end

    # Estimates the relative cost of {.correlate_basic}.
    # Returns a comparative cost without executing correlation or allocating its
    # result. The estimate includes preparation and is not a timing measurement.
    # Compare with {.predict_correlate_fft_time} for the same inputs/options.
    # @param (see convolve)
    # @return [Float] estimated relative cost on the selected platform profile
    # @raise (see predict_convolve_basic_time)
    # @example Estimate direct correlation work
    #   cost = Convolver.predict_correlate_basic_time(Numo::DFloat.ones(32), Numo::DFloat.ones(3))
    #   cost.finite? && cost >= 0 # => true
    # @see file:docs/rules.md Interpretation and limits of estimates
    def predict_correlate_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                     fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic_time
    end

    private

    private :convolve_basic_valid, :correlate_basic_valid, :fft_fast_shape, :fft_shape_cost

    def execution(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
      OperationExecution.new(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
    end
  end
end
