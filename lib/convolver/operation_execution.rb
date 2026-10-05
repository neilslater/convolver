# frozen_string_literal: true

module Convolver
  # Coordinates validation, implementation selection, and one operation family.
  # @private
  class OperationExecution
    DIRECT_OPERATION_COST = { 1 => 3.8e-10, 2 => 5.9e-10, 3 => 4.5e-10 }.freeze
    DOUBLE_OPERATION_COST = { 1 => 3.9e-10, 2 => 6.0e-10, 3 => 4.6e-10 }.freeze
    LARGE_KERNEL_COST = 6.5e-10
    FFT_SPEEDUP_MARGIN = 0.8

    def initialize(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @mode = mode
      @boundary = boundary
      @fill_value = fill_value
      @origin = origin
      @dtype = dtype
    end

    def automatic
      return basic if plan.result_shape.empty?

      threshold = FFT_SPEEDUP_MARGIN * basic_time
      return basic if FftCost.lower_bound(plan, kernel.shape, threshold:) >= threshold
      return fft if automatic_fft_time < threshold

      basic
    end

    def basic
      plan.validate_basic!
      prepared_signal, prepared_kernel = prepared_inputs
      extended_signal = plan.extend_signal(prepared_signal)
      return Convolver.send(:correlate_basic_valid, extended_signal, prepared_kernel) if operation == :correlation

      Convolver.send(:convolve_basic_valid, extended_signal, prepared_kernel)
    end

    def fft
      plan.fft_buffers
      return prepared_inputs.reduce(:*) if signal.ndim.zero?

      fft_operation.call
    end

    def fft_time
      FftEstimator.new(operation, signal, kernel, plan).call
    end

    def basic_time
      plan.validate_basic!
      calculation_cost = direct_operation_cost * plan.result_size * kernel.size
      calculation_cost + preparation_cost
    end

    private

    attr_reader :operation, :signal, :kernel, :mode, :boundary, :fill_value, :origin, :dtype

    def plan
      @plan ||= OperationPlan.new(signal, kernel, operation:, mode:, boundary:, fill_value:, origin:, dtype:)
    end

    def prepared_inputs
      @prepared_inputs ||= [plan.dtype.cast(signal), plan.dtype.cast(kernel)]
    end

    def fft_operation
      prepared_signal, prepared_kernel = prepared_inputs
      return CircularFftOperation.new(operation, prepared_signal, prepared_kernel, plan) if plan.wrap?

      LinearFftOperation.new(operation, plan.extend_signal(prepared_signal), prepared_kernel, plan)
    end

    def automatic_fft_time
      fft_time
    rescue FftUnavailable
      Float::INFINITY
    end

    def preparation_cost
      costs = PreparationCost.new(plan)
      costs.conversion + costs.extension
    end

    def direct_operation_cost
      return LARGE_KERNEL_COST if kernel.size > 64 && signal.ndim != 2

      costs = plan.dtype == Numo::DFloat ? DOUBLE_OPERATION_COST : DIRECT_OPERATION_COST
      costs.fetch(signal.ndim, costs.values.last)
    end
  end

  private_constant :OperationExecution
end
