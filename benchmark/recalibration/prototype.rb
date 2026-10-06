# frozen_string_literal: true

require_relative 'profiles'
require_relative 'cost_model'

module Recalibration
  # Isolated subclass: same preparation, preflight and native/FFT execution as production.
  class Prototype < Convolver.const_get(:OperationExecution)
    def self.call(operation, signal, kernel, mode: :valid, boundary: :constant,
                  fill_value: Convolver.const_get(:UNSPECIFIED_FILL), origin: 0, dtype: nil)
      execution(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:).automatic
    end

    # Mirror the production public wrapper and its private execution factory.
    def self.execution(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
      new(operation, signal, kernel, mode:, boundary:, fill_value:, origin:, dtype:)
    end
    private_class_method :execution

    def automatic
      return super unless calibrated?

      threshold = fft_margin * basic_time
      return basic if cost_model.lower_bound(threshold) >= threshold
      return fft if automatic_fft_time < threshold

      basic
    end

    def basic_time
      calibrated? ? cost_model.basic_time : super
    end

    def fft_time
      calibrated? ? cost_model.fft_time : super
    end

    private

    def fft_margin
      FFT_SPEEDUP_MARGIN
    end

    def calibrated?
      Profiles::CURRENT && (1..3).cover?(plan.result_shape.length)
    end

    def cost_model
      @cost_model ||= CostModel.new(operation, signal, kernel, plan, Profiles::CURRENT)
    end
  end

  # One controlled policy variation, with identical frozen coefficients.
  class MarginPrototype < Prototype
    private

    def fft_margin
      0.9
    end
  end
end
