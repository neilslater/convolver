# frozen_string_literal: true

require 'convolver/fft_cost'

module Convolver
  # Estimates the selected real or complex PocketFFT implementation cost.
  # @private
  class FftEstimator
    AXIS_MOVE_COST = 1.6e-9
    RESULT_COST = 5.0e-10
    CORRELATION_COST = 2.0e-10

    def initialize(operation, signal, kernel, plan)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @plan = plan
      @buffers = plan.fft_buffers
      @preparation = PreparationCost.new(plan)
    end

    def call
      return FftCost.fixed(0) + preparation.conversion + result_cost if signal.ndim.zero?

      transform_cost + preparation_cost + extra_cost
    end

    private

    attr_reader :operation, :signal, :kernel, :plan, :buffers, :preparation

    def transform_cost
      transform = FftCost.transform(buffers.transform_size, signal.ndim, real: !buffers.real_axis.nil?)
      FftCost.fixed(signal.ndim) + transform + (AXIS_MOVE_COST * moved_size)
    end

    def extra_cost
      correlation_cost = operation == :correlation ? CORRELATION_COST * buffers.spectrum_size : 0.0
      correlation_cost + preparation.conversion + result_cost
    end

    def result_cost
      RESULT_COST * plan.result_size * (plan.dtype::ELEMENT_BYTE_SIZE / 4)
    end

    def preparation_cost
      return circular_preparation_cost if plan.wrap?

      (2 * FftCost::PREPARATION * buffers.transform_size) + preparation.extension
    end

    def circular_preparation_cost
      (FftCost::PREPARATION * (signal.size + kernel.size)) + preparation.folding(kernel.size)
    end

    def moved_size
      buffers.real_axis && buffers.real_axis != signal.ndim - 1 ? 3 * signal.size : 0
    end
  end

  private_constant :FftEstimator
end
