# frozen_string_literal: true

require 'numo/narray/alt'
require 'numo/pocketfft'
require 'convolver/convolver'
require 'convolver/version'

# Valid cross-correlation operations for Numo::NArray values.
module Convolver
  # Maximum number of dimensions supported by the direct native implementation.
  MAX_RANK = 16

  class << self
    # Chooses the likely fastest implementation for a valid cross-correlation.
    #
    # The inputs must have the same rank, and the kernel must not be larger than
    # the signal in any dimension. The result shape is:
    #
    #   signal.shape.zip(kernel.shape).map { |signal_size, kernel_size| signal_size - kernel_size + 1 }
    #
    # @param signal [Numo::NArray] input values
    # @param kernel [Numo::NArray] correlation kernel
    # @return [Numo::SFloat] valid cross-correlation result
    # @raise [ArgumentError] if the inputs have incompatible ranks or shapes
    def convolve(signal, kernel)
      validate_inputs!(signal, kernel)
      return convolve_basic(signal, kernel) if signal.size < 1000 || kernel.size < 100

      basic_time_predicted = predict_convolve_basic_time(signal, kernel)
      return convolve_basic(signal, kernel) if basic_time_predicted < 0.1

      fft_time_predicted = predict_convolve_fft_time(signal, kernel)
      return convolve_fft(signal, kernel) if fft_time_predicted < 2 * basic_time_predicted

      convolve_basic(signal, kernel)
    end

    # Uses PocketFFT to calculate a valid cross-correlation.
    #
    # @param signal [Numo::NArray] input values
    # @param kernel [Numo::NArray] correlation kernel
    # @return [Numo::SFloat] valid cross-correlation result
    # @raise [ArgumentError] if the inputs have incompatible ranks or shapes
    def convolve_fft(signal, kernel)
      validate_inputs!(signal, kernel)
      ranges = kernel.shape.zip(signal.shape).map { |kernel_size, signal_size| (kernel_size - 1)...signal_size }
      full_convolution = Numo::Pocketfft.fftconvolve(signal, kernel.reverse)

      Numo::SFloat.cast(full_convolution[*ranges])
    end

    # Compatibility alias for the former FFTW3-backed implementation.
    #
    # @deprecated Use {.convolve_fft}; Convolver no longer uses FFTW3.
    # @return [Numo::SFloat] valid cross-correlation result
    def convolve_fftw3(signal, kernel)
      warn 'Convolver.convolve_fftw3 is deprecated; use .convolve_fft instead', uplevel: 1
      convolve_fft(signal, kernel)
    end

    # Estimates the relative cost of {.convolve_fft}.
    #
    # @param signal [Numo::NArray] input values
    # @param kernel [Numo::NArray] correlation kernel
    # @return [Float] machine-specific relative cost estimate
    def predict_convolve_fft_time(signal, kernel)
      validate_inputs!(signal, kernel)
      output_size = result_shape(signal.shape, kernel.shape).inject(:*)
      16 * 4.55e-08 * output_size * Math.log(output_size)
    end

    # Estimates the relative cost of {.convolve_basic}.
    #
    # @param signal [Numo::NArray] input values
    # @param kernel [Numo::NArray] correlation kernel
    # @return [Float] machine-specific relative cost estimate
    def predict_convolve_basic_time(signal, kernel)
      validate_inputs!(signal, kernel)
      outputs = result_shape(signal.shape, kernel.shape).inject(:*)
      4.54e-12 * (outputs * signal.size * kernel.size)
    end

    private

    def result_shape(signal_shape, kernel_shape)
      signal_shape.zip(kernel_shape).map { |signal_size, kernel_size| signal_size - kernel_size + 1 }
    end

    def validate_inputs!(signal, kernel)
      validate_types!(signal, kernel)
      validate_shapes!(signal, kernel)
    end

    def validate_types!(signal, kernel)
      unless signal.is_a?(Numo::NArray) && kernel.is_a?(Numo::NArray)
        raise ArgumentError, 'signal and kernel must be Numo::NArray values'
      end
      raise ArgumentError, 'signal and kernel must not be empty' if signal.empty? || kernel.empty?
    end

    def validate_shapes!(signal, kernel)
      raise ArgumentError, 'signal and kernel must have equal rank' unless signal.ndim == kernel.ndim
      raise ArgumentError, "maximum supported rank is #{MAX_RANK}" if signal.ndim > MAX_RANK
      return if signal.shape.zip(kernel.shape).all? { |signal_size, kernel_size| signal_size >= kernel_size }

      raise ArgumentError, 'kernel must not be larger than signal in any dimension'
    end
  end
end
