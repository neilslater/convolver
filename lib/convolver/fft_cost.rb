# frozen_string_literal: true

require 'convolver/preparation_cost'

module Convolver
  # Shared transform costs and a lower bound that never constructs an FFT plan.
  # @private
  class FftCost
    REAL = { 1 => 1.0e-9, 2 => 1.0e-9, 3 => 1.5e-9 }.freeze
    COMPLEX = { 1 => 2.4e-9, 2 => 2.4e-9, 3 => 2.8e-9 }.freeze
    FIXED = { 0 => 5.0e-6, 1 => 1.5e-5, 2 => 3.0e-5, 3 => 1.0e-4 }.freeze
    PREPARATION = 1.0e-9

    def self.fixed(rank, profile: nil)
      return profile.fetch(:fixed).fetch(rank - 1) if profile

      FIXED.fetch(rank, FIXED.values.last)
    end

    def self.transform(size, rank, real: true, profile: nil)
      coefficient(rank, real:, profile:) * size * Math.log([size, 2].max)
    end

    def self.lower_bound(plan, kernel_shape, threshold:)
      rank = kernel_shape.length
      profile = plan.cost_profile
      setup = fixed(rank, profile:)
      return setup if setup >= threshold || plan.wrap? || rank.zero?

      size = minimum_size(plan.extended_shape, kernel_shape)
      setup + transform(size, rank, profile:) + (2 * PREPARATION * size) + PreparationCost.new(plan).indices
    end

    def self.coefficient(rank, real:, profile:)
      return profile.fetch(:transform).fetch(rank - 1) * (real ? 1 : 2.2) if profile

      costs = real ? REAL : COMPLEX
      costs.fetch(rank, costs.values.last)
    end
    private_class_method :coefficient

    def self.minimum_size(signal_shape, kernel_shape)
      # Ruby integers avoid overflow: this is only a cost bound, never an allocation.
      shape = signal_shape.zip(kernel_shape).map { |signal, kernel| signal + kernel - 1 }
      shape[-1] += 1 if shape[-1].odd?
      shape.reduce(1, :*)
    end
    private_class_method :minimum_size
  end

  private_constant :FftCost
end
