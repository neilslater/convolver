# frozen_string_literal: true

module Convolver
  # Calculates a valid linear operation through real PocketFFT transforms.
  class LinearFftOperation
    def initialize(operation, signal, kernel, plan)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @transform_shape = plan.linear_fft_shape(kernel.shape)
    end

    def call
      signal_padded = zero_padded(signal)
      kernel_padded = zero_padded(kernel)
      full_result = real_spectrum_product(signal_padded, kernel_padded)
      Numo::SFloat.cast(full_result[*valid_ranges])
    end

    private

    attr_reader :operation, :signal, :kernel, :transform_shape

    def zero_padded(value)
      Numo::DFloat.zeros(*transform_shape).tap do |padded|
        padded[*value.shape.map { |size| 0...size }] = value
      end
    end

    def real_spectrum_product(signal_value, kernel_value)
      signal_spectrum = Numo::Pocketfft.rfftn(signal_value)
      kernel_spectrum = Numo::Pocketfft.rfftn(kernel_value)
      kernel_spectrum.inplace.conj if operation == :correlation
      Numo::Pocketfft.irfftn(signal_spectrum.inplace * kernel_spectrum)
    end

    def valid_ranges
      signal.shape.zip(kernel.shape).map do |signal_size, kernel_size|
        if operation == :correlation
          0...(signal_size - kernel_size + 1)
        else
          (kernel_size - 1)...signal_size
        end
      end
    end
  end

  private_constant :LinearFftOperation
end
