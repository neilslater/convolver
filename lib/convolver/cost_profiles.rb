# frozen_string_literal: true

module Convolver
  # Immutable measured-environment coefficients in seconds; ranks one through three.
  # @private
  module CostProfiles
    MAC = {
      direct: [[6.55e-6, 3.24e-10, 7.08e-10].freeze, [6.55e-6, 3.24e-10, 7.08e-10].freeze].freeze,
      fixed: [1.64e-5, 5.12e-5, 1.13e-4].freeze,
      transform: [8.02e-10, 7.17e-10, 1.54e-9].freeze,
      indices: { reflect: 1.33e-7, nearest: 1.46e-7, mirror: 1.57e-7, wrap: 3.23e-8 }.freeze,
      folding: [3.78e-7, 3.99e-7, 5.09e-7].freeze
    }.freeze
    X86 = {
      direct: [[1.59e-5, 3.08e-10, 2.52e-10].freeze, [1.63e-5, 3.22e-10, 3.96e-10].freeze].freeze,
      fixed: [3.77e-5, 1.20e-4, 1.98e-4].freeze,
      transform: [2.00e-9, 2.13e-9, 3.13e-9].freeze,
      indices: { reflect: 2.32e-7, nearest: 2.88e-7, mirror: 2.62e-7, wrap: 1.11e-7 }.freeze,
      folding: [8.20e-7, 7.51e-7, 9.79e-7].freeze
    }.freeze
    ARM = {
      direct: [[1.88e-5, 5.12e-10, 5.96e-10].freeze, [1.88e-5, 5.12e-10, 5.96e-10].freeze].freeze,
      fixed: [4.20e-5, 1.20e-4, 2.70e-4].freeze,
      transform: [3.29e-9, 2.89e-9, 3.74e-9].freeze,
      indices: { reflect: 2.62e-7, nearest: 3.67e-7, mirror: 2.87e-7, wrap: 1.50e-7 }.freeze,
      folding: [9.97e-7, 8.80e-7, 1.24e-6].freeze
    }.freeze
    def self.for_platform(platform)
      case platform
      when /\Aarm64-darwin/ then MAC
      when /\Ax86_64-linux/ then X86
      when /\Aaarch64-linux/ then ARM
      end
    end

    CURRENT = for_platform(RUBY_PLATFORM)

    def self.for_rank(rank)
      CURRENT if (1..3).cover?(rank)
    end
  end

  private_constant :CostProfiles
end
