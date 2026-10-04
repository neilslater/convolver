# frozen_string_literal: true

require 'convolver/fft_limits'

module Convolver
  # Preflights casts, extension, indexing and direct-path cache allocations.
  # @private
  class BufferPreparation
    def initialize(plan, signal_shape, kernel_shape, limits: BufferLimits::NATIVE)
      @plan = plan
      @signal_size = limits.product!(signal_shape, 'signal size')
      @kernel_size = limits.product!(kernel_shape, 'kernel size')
      @limits = limits
      @scalar = signal_shape.empty?
    end

    def basic!
      common!
      extension!
      limits.buffer!(kernel_size, BufferLimits::INDEX_BYTES, 'direct kernel offset cache')
    end

    def fft!
      common!
      return if @scalar
      return extension! unless plan.wrap?

      limits.buffer!(signal_size, 8, 'circular signal preparation', error: FftUnavailable)
      limits.buffer!(kernel_size, 8, 'circular kernel preparation', error: FftUnavailable)
      limits.buffer!(kernel_size, BufferLimits::INDEX_BYTES, 'circular kernel index buffer', error: FftUnavailable)
    end

    private

    attr_reader :plan, :signal_size, :kernel_size, :limits

    def common!
      limits.buffer!(plan.result_size, 4, 'result buffer')
      limits.buffer!(signal_size, 4, 'signal conversion buffer')
      limits.buffer!(kernel_size, 4, 'kernel conversion buffer')
    end

    def extension!
      return if plan.valid?

      limits.buffer!(plan.extended_size, 4, 'extended signal buffer')
      return if plan.boundary == :constant

      plan.extended_shape.each do |length|
        limits.buffer!(length, BufferLimits::INDEX_BYTES, 'boundary index buffer')
      end
    end
  end

  private_constant :BufferPreparation
end
