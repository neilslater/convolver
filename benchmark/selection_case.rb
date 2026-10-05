# frozen_string_literal: true

require_relative 'selection_measurement'

module SelectionBenchmark
  # One input pair, operation and dtype measured through all three public paths.
  class Case
    def initialize(definition, type, operation, view: false)
      @name, @signal_shape, @kernel_shape, @options = definition
      @type = type
      @operation = operation
      @view = view
      @signal = input(@signal_shape, 17, 8)
      @kernel = input(@kernel_shape, 7, 3)
    end

    def run
      calls = { basic: invocation("#{@operation}_basic"), fft: invocation("#{@operation}_fft"),
                automatic: invocation(@operation) }
      validate(calls)
      metadata.merge(estimates).merge(Measurement.new(calls).run)
    end

    private

    def input(shape, period, offset)
      value = (@type.new(*shape).seq % period) - offset
      value = value.reverse(0) if @view
      value.freeze
    end

    def invocation(method)
      -> { Convolver.public_send(method, @signal, @kernel, **@options) }
    end

    def metadata
      { name: @name, type: @type.name, operation: @operation, view: @view,
        signal: @signal_shape, kernel: @kernel_shape, options: @options }
    end

    def estimates
      { basic_estimate_us: invocation("predict_#{@operation}_basic_time").call * 1e6,
        fft_estimate_us: invocation("predict_#{@operation}_fft_time").call * 1e6 }
    end

    def validate(calls)
      reference = calls.fetch(:basic).call
      %i[fft automatic].each { |method| validate_result(reference, calls.fetch(method).call) }
    end

    def validate_result(reference, result)
      error = (result - reference).abs.max
      matches = result.instance_of?(reference.class) && result.shape == reference.shape
      return if matches && error.finite? && error <= [1.0, reference.abs.max].max * 1e-5

      raise "Result mismatch: #{metadata}, error=#{error}"
    end
  end
end
