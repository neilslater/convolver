# frozen_string_literal: true

require_relative '../selection_measurement'

module Recalibration
  # Prepared-input probes isolate work; they are not substitutes for complete-call timings.
  class Components
    def initialize(operation, signal, kernel, plan, plan_factory)
      @operation = operation
      @plan = plan
      @signal = plan.dtype.cast(signal)
      @kernel = plan.dtype.cast(kernel)
      @plan_factory = plan_factory
    end

    def run
      return {} if @signal.ndim.zero?

      SelectionBenchmark::Measurement.new(calls, samples: 3, duration: 0.002, maximum: 100).run
    end

    private

    def calls
      extended = @plan.extend_signal(@signal)
      method = @operation == :correlation ? :correlate_basic_valid : :convolve_basic_valid
      { plan: @plan_factory, fft_plan: -> { @plan_factory.call.fft_buffers },
        extension: -> { @plan.extend_signal(@signal) },
        native_direct: -> { Convolver.send(method, extended, @kernel) } }.merge(fft_calls(extended))
    end

    def fft_calls(extended)
      if @plan.wrap?
        factory = -> { Convolver.const_get(:CircularFftOperation).new(@operation, @signal, @kernel, @plan) }
        operation = factory.call
        { folding_preparation: factory, fft_execution: operation.method(:call) }
      else
        operation = Convolver.const_get(:LinearFftOperation).new(@operation, extended, @kernel, @plan)
        { fft_execution: operation.method(:call) }
      end
    end
  end
end
