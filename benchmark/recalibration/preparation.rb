# frozen_string_literal: true

module Recalibration
  # Retains existing copy costs, with measured Ruby index and folding rates.
  class Preparation < Convolver.const_get(:PreparationCost)
    def initialize(plan, profile)
      super(plan)
      @profile = profile
    end

    def indices
      return 0.0 if plan.valid? || plan.boundary == :constant

      @profile.fetch(:indices).fetch(plan.boundary) * plan.extended_shape.sum
    end

    def folding(kernel_size)
      @profile.fetch(:folding).fetch(plan.result_shape.length - 1) * kernel_size
    end
  end
end
