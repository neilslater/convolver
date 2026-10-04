# frozen_string_literal: true

require 'convolver/fft_limits'
require 'convolver/buffer_preparation'

module Convolver
  # Shares allocation-free FFT dimensions between execution and estimation.
  # @private
  class FftBufferPlan
    attr_reader :transform_shape, :real_axis, :transform_size, :spectrum_size

    def initialize(plan, signal_shape, kernel_shape, limits: BufferLimits::NATIVE)
      BufferPreparation.new(plan, signal_shape, kernel_shape, limits:).fft!
      @real_axis = plan.wrap? ? CircularFftOperation.real_axis(signal_shape) : signal_shape.length - 1
      @transform_shape = select_shape(plan, signal_shape, kernel_shape, FftLimits.new(limits))
      @transform_size = transform_shape.reduce(1, :*)
      @spectrum_size = spectrum_elements
    end

    private

    def select_shape(plan, signal_shape, kernel_shape, limits)
      return [].freeze if signal_shape.empty?
      return plan.linear_fft_shape(kernel_shape, limits:) unless plan.wrap?

      limits.validate!(signal_shape, real_axis:)
      signal_shape.dup.freeze
    end

    def spectrum_elements
      return transform_size if transform_shape.empty? || real_axis.nil?

      length = transform_shape[real_axis]
      transform_size / length * ((length / 2) + 1)
    end
  end

  private_constant :FftBufferPlan
end
