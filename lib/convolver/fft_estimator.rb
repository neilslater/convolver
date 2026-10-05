# frozen_string_literal: true

module Convolver
  # Estimates the selected real or complex PocketFFT implementation cost.
  # @private
  class FftEstimator
    REAL_FFT_COST = { 1 => 1.0e-9, 2 => 9.0e-10, 3 => 1.45e-9 }.freeze
    COMPLEX_FFT_COST = { 1 => 2.4e-9, 2 => 2.4e-9, 3 => 2.8e-9 }.freeze
    FIXED_COST = 1.5e-4
    PREPARATION_COST = 1.0e-9
    AXIS_MOVE_COST = 1.6e-9
    CONVERSION_COST = 7.5e-10
    RESULT_COST = 5.0e-10
    CORRELATION_COST = 2.0e-10

    def initialize(operation, signal, kernel, plan)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @plan = plan
      @buffers = plan.fft_buffers
    end

    def call
      transform_size, spectrum_size, coefficient, moved_size = dimensions
      transform_cost = coefficient * transform_size * Math.log([transform_size, 2].max)
      preparation_cost = PREPARATION_COST * preparation_size(transform_size)
      axis_move_cost = AXIS_MOVE_COST * moved_size
      FIXED_COST + transform_cost + preparation_cost + axis_move_cost + extra_cost(spectrum_size)
    end

    private

    attr_reader :operation, :signal, :kernel, :plan, :buffers

    def dimensions
      return linear_dimensions unless plan.wrap?

      circular_dimensions
    end

    def circular_dimensions
      real_axis = buffers.real_axis
      [buffers.transform_size, buffers.spectrum_size, transform_coefficient(real_axis), moved_size(real_axis)]
    end

    def linear_dimensions
      [buffers.transform_size, buffers.spectrum_size,
       cost_for(REAL_FFT_COST), 0]
    end

    def extra_cost(spectrum_size)
      correlation_cost = operation == :correlation ? CORRELATION_COST * spectrum_size : 0.0
      correlation_cost + (CONVERSION_COST * plan.conversion_size) + result_cost
    end

    def result_cost
      RESULT_COST * plan.result_size * (plan.dtype::ELEMENT_BYTE_SIZE / 4)
    end

    def preparation_size(transform_size)
      extension_size = extension_elements
      working_size = plan.wrap? ? signal.size + kernel.size : 2 * transform_size
      extension_size + working_size
    end

    def extension_elements
      return 0 if plan.valid? || plan.wrap?

      plan.extended_size * (plan.dtype::ELEMENT_BYTE_SIZE / 4)
    end

    def cost_for(costs)
      costs.fetch(signal.ndim, costs.values.last)
    end

    def transform_coefficient(real_axis)
      cost_for(real_axis ? REAL_FFT_COST : COMPLEX_FFT_COST)
    end

    def moved_size(real_axis)
      real_axis && real_axis != signal.ndim - 1 ? 3 * signal.size : 0
    end
  end

  private_constant :FftEstimator
end
