# frozen_string_literal: true

require_relative 'preparation'

module Recalibration
  # Research-only cost model; public production estimators remain unchanged.
  class CostModel
    def initialize(operation, signal, kernel, plan, profile)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @plan = plan
      @profile = profile
      @preparation = Preparation.new(plan, profile)
    end

    def basic_time
      @plan.validate_basic!
      direct_work + @preparation.conversion + @preparation.extension
    end

    def fft_time
      buffers = @plan.fft_buffers
      work = transform(buffers.transform_size, real: !buffers.real_axis.nil?)
      setup + work + fft_preparation(buffers) + extras(buffers)
    end

    def lower_bound(threshold)
      return setup if setup >= threshold || @plan.wrap?

      size = Convolver.const_get(:FftCost).send(:minimum_size, @plan.extended_shape, @kernel.shape)
      setup + transform(size) + (2e-9 * size) + @preparation.indices
    end

    private

    def direct_work
      fixed, short, long = @profile.fetch(:direct).fetch(@plan.dtype == Numo::DFloat ? 1 : 0)
      work = (short * [@kernel.size, 64].min) + (long * [@kernel.size - 64, 0].max)
      fixed + (work * @plan.result_size)
    end

    def setup
      @profile.fetch(:fixed).fetch(@signal.ndim - 1)
    end

    def transform(size, real: true)
      @profile.fetch(:transform).fetch(@signal.ndim - 1) * size * Math.log([size, 2].max) * (real ? 1 : 2.2)
    end

    def fft_preparation(buffers)
      return (2e-9 * buffers.transform_size) + @preparation.extension unless @plan.wrap?

      (1e-9 * (@signal.size + @kernel.size)) + @preparation.folding(@kernel.size)
    end

    def extras(buffers)
      result = 5e-10 * @plan.result_size * (@plan.dtype::ELEMENT_BYTE_SIZE / 4)
      correlation = @operation == :correlation ? 2e-10 * buffers.spectrum_size : 0.0
      @preparation.conversion + result + correlation + movement(buffers)
    end

    def movement(buffers)
      moved = buffers.real_axis && buffers.real_axis != @signal.ndim - 1 ? 3 * @signal.size : 0
      1.6e-9 * moved
    end
  end
end
