# frozen_string_literal: true

module Convolver
  # Coordinates validation, implementation selection, and one operation family.
  class OperationExecution
    DIRECT_OPERATION_COST = { 1 => 6.8e-10, 2 => 5.9e-10, 3 => 4.5e-10 }.freeze
    EXTENSION_COST = 1.0e-9
    FFT_SPEEDUP_MARGIN = 0.8

    METHODS = {
      correlation: {
        basic: :correlate_basic,
        fft: :correlate_fft,
        predict_basic: :predict_correlate_basic_time,
        predict_fft: :predict_correlate_fft_time
      },
      convolution: {
        basic: :convolve_basic,
        fft: :convolve_fft,
        predict_basic: :predict_convolve_basic_time,
        predict_fft: :predict_convolve_fft_time
      }
    }.freeze

    def initialize(operation, signal, kernel, mode:, boundary:, fill_value:, origin:)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @mode = mode
      @boundary = boundary
      @fill_value = fill_value
      @origin = origin
    end

    def automatic(receiver)
      return invoke(receiver, :basic) if plan.extended_size < 1000

      direct_time = invoke(receiver, :predict_basic)
      return invoke(receiver, :fft) if invoke(receiver, :predict_fft) < FFT_SPEEDUP_MARGIN * direct_time

      invoke(receiver, :basic)
    end

    def basic
      extended_signal = plan.extend_signal(signal)
      return Convolver.send(:correlate_basic_valid, extended_signal, kernel) if operation == :correlation

      Convolver.send(:convolve_basic_valid, extended_signal, kernel)
    end

    def fft
      plan
      return Numo::SFloat.cast(signal) * Numo::SFloat.cast(kernel) if signal.ndim.zero?

      fft_operation.call
    end

    def fft_time
      FftEstimator.new(operation, signal, kernel, plan).call
    end

    def basic_time
      calculation_cost = direct_operation_cost * plan.result_size * kernel.size
      extension_cost = plan.valid? ? 0.0 : EXTENSION_COST * plan.extended_size
      calculation_cost + extension_cost
    end

    private

    attr_reader :operation, :signal, :kernel, :mode, :boundary, :fill_value, :origin

    def plan
      @plan ||= OperationPlan.new(signal, kernel, operation:, mode:, boundary:, fill_value:, origin:)
    end

    def invoke(receiver, method)
      receiver.public_send(METHODS.fetch(operation).fetch(method), signal, kernel, **public_options)
    end

    def public_options
      options = { mode:, boundary:, origin: }
      options[:fill_value] = fill_value unless fill_value.equal?(UNSPECIFIED_FILL)
      options
    end

    def fft_operation
      return CircularFftOperation.new(operation, signal, kernel, plan) if plan.wrap?

      LinearFftOperation.new(operation, plan.extend_signal(signal), kernel, plan)
    end

    def direct_operation_cost
      DIRECT_OPERATION_COST.fetch(signal.ndim, DIRECT_OPERATION_COST.values.last)
    end
  end

  private_constant :OperationExecution
end
