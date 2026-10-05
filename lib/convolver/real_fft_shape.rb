# frozen_string_literal: true

require 'convolver/fft_limits'

module Convolver
  # Selects a safe real-transform shape from exact/even and fast candidates.
  # @private
  class RealFftShape
    def initialize(minimum_shape, size_max:, limits: nil)
      @minimum_shape = minimum_shape
      @size_max = size_max
      @limits = limits
    end

    def call
      exact_shape = even_final_axis(minimum_shape)
      candidates = [exact_shape, fast_shape(exact_shape)].compact.select { |shape| admissible?(shape) }
      (candidates.min_by { |shape| transform_cost(shape) } || raise_rejection).freeze
    end

    private

    attr_reader :minimum_shape, :size_max, :limits

    def admissible?(shape)
      return false unless representable?(shape)

      limits&.validate!(shape, real_axis: shape.length - 1)
      true
    rescue FftUnavailable => e
      @rejection ||= e
      false
    end

    def raise_rejection
      raise @rejection if @rejection

      raise_overflow
    end

    def fast_shape(exact_shape)
      Convolver.send(:fft_fast_shape, exact_shape, size_max)
    end

    def even_final_axis(shape)
      result = shape.dup
      result[-1] = checked_add(result[-1], 1) if result[-1].odd?
      result.freeze
    end

    def transform_cost(shape)
      Convolver.send(:fft_shape_cost, shape)
    end

    def representable?(shape)
      shape.reduce(1) do |product, size|
        return false if !product.zero? && size > size_max / product

        product * size
      end
      true
    end

    def checked_add(left, right)
      return left + right if right <= size_max - left

      raise_overflow
    end

    def raise_overflow
      raise(limits ? FftUnavailable : RangeError, 'FFT shape exceeds native implementation limit')
    end
  end

  private_constant :RealFftShape
end
