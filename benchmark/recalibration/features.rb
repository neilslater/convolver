# frozen_string_literal: true

module Recalibration
  # Untimed work counts for diagnosing the existing public estimates.
  class Features
    def initialize(signal, kernel, plan)
      @signal = signal
      @kernel = kernel
      @plan = plan
      @buffers = plan.fft_buffers
    end

    def capture
      dimensions.merge(preparation).merge(transform)
    end

    private

    def dimensions
      { rank: @signal.ndim, dtype: @plan.dtype.name, signal_size: @signal.size, kernel_size: @kernel.size,
        result_size: @plan.result_size, operations: @plan.result_size * @kernel.size,
        extended_shape: @plan.extended_shape, transform_shape: @buffers.transform_shape }
    end

    def preparation
      { conversion: @plan.conversion_size, width: @plan.dtype::ELEMENT_BYTE_SIZE / 4,
        extension: @plan.valid? ? 0 : @plan.extended_size,
        indices: @plan.valid? || @plan.boundary == :constant ? 0 : @plan.extended_shape.sum,
        boundary: @plan.boundary, wrap: @plan.wrap?, folding: @plan.wrap? ? @kernel.size : 0 }
    end

    def transform
      { transform_size: @buffers.transform_size, spectrum_size: @buffers.spectrum_size,
        real: !@buffers.real_axis.nil?, moved: moved_size }
    end

    def moved_size
      @buffers.real_axis && @buffers.real_axis != @signal.ndim - 1 ? 3 * @signal.size : 0
    end
  end
end
