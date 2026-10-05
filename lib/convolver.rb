# frozen_string_literal: true

require 'numo/narray/alt'
require 'numo/pocketfft'
require 'convolver/convolver'
require 'convolver/version'

# Mathematical convolution and cross-correlation for Numo::NArray values.
module Convolver
  # Maximum number of dimensions supported by the implementations.
  MAX_RANK = 16

  require 'convolver/operation_plan'
  require 'convolver/linear_fft_operation'
  require 'convolver/circular_fft_operation'
  require 'convolver/fft_estimator'
  require 'convolver/operation_execution'

  class << self
    # Chooses the likely fastest mathematical convolution implementation.
    #
    # Results use the dtype selected from both inputs, unless dtype overrides it.
    # Inputs must be nonempty arrays of equal rank, with at most {MAX_RANK}
    # dimensions. Scalars multiply directly.
    # In valid mode the kernel must fit inside the signal in every dimension.
    # Full mode permits only constant boundaries; nonconstant boundaries and
    # nonzero origins are supported only in same mode.
    #
    # @param signal [Numo::NArray] input signal
    # @param kernel [Numo::NArray] convolution kernel
    # @param mode [:valid, :same, :full] output extent (default: :valid)
    # @param boundary [:constant, :nearest, :reflect, :mirror, :wrap]
    #   signal extension policy (default: :constant)
    # @param fill_value [Numeric] constant boundary value (default: 0);
    #   valid mode requires zero, and other boundaries reject this option
    # @param origin [Integer, Array<Integer>] offset from the kernel midpoint,
    #   shared across axes or one per axis (default: 0)
    # @param dtype [Class, nil] Numo::SFloat or Numo::DFloat; nil promotes both inputs.
    #   Integer types of 8/16 bits select SFloat; 32/64 bits select DFloat.
    #   Only these concrete integer and floating classes are supported.
    # @return [Numo::SFloat, Numo::DFloat] mathematical convolution result
    # @raise [ArgumentError] if inputs or options are invalid
    # @raise [RangeError] if planned dimensions, buffers or FFT arithmetic exceed native limits
    def convolve(signal, kernel, mode: :valid, boundary: :constant,
                 fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).automatic
    end

    # Chooses the likely fastest cross-correlation implementation.
    # Uses the same input and option rules as {.convolve}, without reversing
    # the kernel's orientation.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] cross-correlation result
    # @raise (see convolve)
    def correlate(signal, kernel, mode: :valid, boundary: :constant,
                  fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).automatic
    end

    # Uses the direct native mathematical convolution implementation.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] mathematical convolution result
    # @raise (see convolve)
    def convolve_basic(signal, kernel, mode: :valid, boundary: :constant,
                       fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic
    end

    # Uses the direct native cross-correlation implementation.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] cross-correlation result
    # @raise (see convolve)
    def correlate_basic(signal, kernel, mode: :valid, boundary: :constant,
                        fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic
    end

    # Uses PocketFFT to calculate mathematical convolution.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] mathematical convolution result
    # @raise (see convolve)
    def convolve_fft(signal, kernel, mode: :valid, boundary: :constant,
                     fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft
    end

    # Uses PocketFFT to calculate cross-correlation.
    # @param (see convolve)
    # @return [Numo::SFloat, Numo::DFloat] cross-correlation result
    # @raise (see convolve)
    def correlate_fft(signal, kernel, mode: :valid, boundary: :constant,
                      fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft
    end

    # Estimates the relative cost of {.convolve_fft}.
    # This is a heuristic for algorithm selection, not a timing measurement.
    # @param (see convolve)
    # @return [Float] machine-specific relative cost estimate
    # @raise (see convolve)
    def predict_convolve_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                  fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft_time
    end

    # Estimates the relative cost of {.correlate_fft}.
    # This is a heuristic for algorithm selection, not a timing measurement.
    # @param (see convolve)
    # @return [Float] machine-specific relative cost estimate
    # @raise (see convolve)
    def predict_correlate_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                   fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).fft_time
    end

    # Estimates the relative cost of {.convolve_basic}.
    # This is a heuristic for algorithm selection, not a timing measurement.
    # @param (see convolve)
    # @return [Float] machine-specific relative cost estimate
    # @raise (see convolve)
    def predict_convolve_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                    fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic_time
    end

    # Estimates the relative cost of {.correlate_basic}.
    # This is a heuristic for algorithm selection, not a timing measurement.
    # @param (see convolve)
    # @return [Float] machine-specific relative cost estimate
    # @raise (see convolve)
    def predict_correlate_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                     fill_value: UNSPECIFIED_FILL, origin: 0, dtype: nil)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).basic_time
    end

    private

    private :convolve_basic_valid, :correlate_basic_valid

    def execution(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
      OperationExecution.new(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
    end
  end
end
