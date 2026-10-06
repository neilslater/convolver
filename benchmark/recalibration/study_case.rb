# frozen_string_literal: true

require_relative '../selection_case'
require_relative 'features'
require_relative 'components'

module Recalibration
  # Complete public calls, metadata and separate diagnostic probes for one input pair.
  class StudyCase < SelectionBenchmark::Case
    def run
      methods = calls
      validate(methods)
      metadata.merge(estimates).merge(diagnostics).merge(measure(methods))
    end

    private

    def calls
      { basic: invocation("#{@operation}_basic"), fft: invocation("#{@operation}_fft"),
        automatic: invocation(@operation) }
    end

    def measure(calls)
      allocations = calls.transform_values do |call|
        before = GC.stat(:total_allocated_objects)
        5.times { call.call }
        (GC.stat(:total_allocated_objects) - before) / 5.0
      end
      SelectionBenchmark::Measurement.new(calls, samples: 7, duration: 0.006, maximum: 400).run
                                     .merge(allocations: allocations)
    end

    def diagnostics
      plan = new_plan
      { selected: selected,
        features: Features.new(@signal, @kernel, plan).capture,
        components: Components.new(operation, @signal, @kernel, plan, method(:new_plan)).run }
    end

    def operation
      @operation == :convolve ? :convolution : :correlation
    end

    def new_plan
      options = { mode: :valid, boundary: :constant, fill_value: Convolver.const_get(:UNSPECIFIED_FILL), origin: 0 }
      Convolver.const_get(:OperationPlan).new(@signal, @kernel, operation:, **options.merge(@options))
    end

    def selected
      path = nil
      trace = TracePoint.new(:call) do |event|
        next unless event.defined_class == Convolver.const_get(:OperationExecution)

        path = event.method_id if %i[basic fft].include?(event.method_id)
      end
      trace.enable { invocation(@operation).call }
      path
    end
  end
end
