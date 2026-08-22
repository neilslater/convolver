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
    # @return [Numo::SFloat] mathematical convolution result
    def convolve(signal, kernel, mode: :valid, boundary: :constant,
                 fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:).automatic(self)
    end

    # Chooses the likely fastest cross-correlation implementation.
    # @return [Numo::SFloat] cross-correlation result
    def correlate(signal, kernel, mode: :valid, boundary: :constant,
                  fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:).automatic(self)
    end

    # Uses the direct native mathematical convolution implementation.
    # @return [Numo::SFloat] mathematical convolution result
    def convolve_basic(signal, kernel, mode: :valid, boundary: :constant,
                       fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:).basic
    end

    # Uses the direct native cross-correlation implementation.
    # @return [Numo::SFloat] cross-correlation result
    def correlate_basic(signal, kernel, mode: :valid, boundary: :constant,
                        fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:).basic
    end

    # Uses PocketFFT to calculate mathematical convolution.
    # @return [Numo::SFloat] mathematical convolution result
    def convolve_fft(signal, kernel, mode: :valid, boundary: :constant,
                     fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:).fft
    end

    # Uses PocketFFT to calculate cross-correlation.
    # @return [Numo::SFloat] cross-correlation result
    def correlate_fft(signal, kernel, mode: :valid, boundary: :constant,
                      fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:).fft
    end

    # Estimates the relative cost of {.convolve_fft}.
    # @return [Float] machine-specific relative cost estimate
    def predict_convolve_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                  fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:).fft_time
    end

    # Estimates the relative cost of {.correlate_fft}.
    # @return [Float] machine-specific relative cost estimate
    def predict_correlate_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                   fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:).fft_time
    end

    # Estimates the relative cost of {.convolve_basic}.
    # @return [Float] machine-specific relative cost estimate
    def predict_convolve_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                    fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:convolution, signal, kernel, mode:, boundary:, fill_value:, origin:).basic_time
    end

    # Estimates the relative cost of {.correlate_basic}.
    # @return [Float] machine-specific relative cost estimate
    def predict_correlate_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                     fill_value: UNSPECIFIED_FILL, origin: 0)
      execution(:correlation, signal, kernel, mode:, boundary:, fill_value:, origin:).basic_time
    end

    private

    private :convolve_basic_valid, :correlate_basic_valid

    def execution(operation, signal, kernel, mode:, boundary:, fill_value:, origin:)
      OperationExecution.new(operation, signal, kernel, mode:, boundary:, fill_value:, origin:)
    end
  end
end
