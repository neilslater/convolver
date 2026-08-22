# frozen_string_literal: true

module Convolver
  # Estimates the selected real or complex PocketFFT implementation cost.
  class FftEstimator
    REAL_FFT_COST = { 1 => 1.0e-9, 2 => 9.0e-10, 3 => 1.45e-9 }.freeze
    COMPLEX_FFT_COST = { 1 => 2.4e-9, 2 => 2.4e-9, 3 => 2.8e-9 }.freeze
    FIXED_COST = 1.5e-4
    PREPARATION_COST = 1.0e-9
    AXIS_MOVE_COST = 1.6e-9
    CORRELATION_COST = 2.0e-10

    def initialize(operation, signal, kernel, plan)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @plan = plan
    end

    def call
      transform_size, spectrum_size, coefficient, moved_size = dimensions
      transform_cost = coefficient * transform_size * Math.log([transform_size, 2].max)
      preparation_cost = PREPARATION_COST * preparation_size(transform_size)
      axis_move_cost = AXIS_MOVE_COST * moved_size
      correlation_cost = operation == :correlation ? CORRELATION_COST * spectrum_size : 0.0
      FIXED_COST + transform_cost + preparation_cost + axis_move_cost + correlation_cost
    end

    private

    attr_reader :operation, :signal, :kernel, :plan

    def dimensions
      return linear_dimensions unless plan.wrap?

      circular_dimensions
    end

    def circular_dimensions
      real_axis = CircularFftOperation.real_axis(signal.shape)
      spectrum_size = real_axis ? real_spectrum_size(real_axis) : signal.size
      [signal.size, spectrum_size, transform_coefficient(real_axis), moved_size(real_axis)]
    end

    def linear_dimensions
      [plan.linear_fft_size(kernel.shape), plan.linear_spectrum_size(kernel.shape),
       cost_for(REAL_FFT_COST), 0]
    end

    def real_spectrum_size(real_axis)
      signal.shape.each_with_index.reduce(1) do |size, (axis_size, axis)|
        size * (axis == real_axis ? (axis_size / 2) + 1 : axis_size)
      end
    end

    def preparation_size(transform_size)
      extension_size = plan.valid? || plan.wrap? ? 0 : plan.extended_size
      working_size = plan.wrap? ? signal.size + kernel.size : 2 * transform_size
      extension_size + working_size
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
