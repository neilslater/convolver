# frozen_string_literal: true

module Convolver
  # Shared estimates for input copies and Ruby boundary/folding work.
  # @private
  class PreparationCost
    CONVERSION_COST = 7.5e-10
    EXTENSION_COST = 1.0e-9
    INDEX_COST = { nearest: 1.45e-7, reflect: 1.33e-7, mirror: 1.54e-7, wrap: 1.05e-7 }.freeze
    FOLD_COST = { 1 => 3.3e-7, 2 => 3.7e-7, 3 => 5.5e-7 }.freeze

    def initialize(plan)
      @plan = plan
    end

    def conversion
      CONVERSION_COST * plan.conversion_size
    end

    def extension
      return 0.0 if plan.valid?

      (EXTENSION_COST * plan.extended_size * (plan.dtype::ELEMENT_BYTE_SIZE / 4)) + indices
    end

    def indices
      return 0.0 if plan.valid? || plan.boundary == :constant

      INDEX_COST.fetch(plan.boundary) * plan.extended_shape.sum
    end

    def folding(kernel_size)
      FOLD_COST.fetch(plan.result_shape.length, FOLD_COST.values.last) * kernel_size
    end

    private

    attr_reader :plan
  end

  private_constant :PreparationCost
end
