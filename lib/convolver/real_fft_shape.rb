# frozen_string_literal: true

module Convolver
  # Selects a safe real-transform shape from exact/even and fast candidates.
  class RealFftShape
    def initialize(minimum_shape, size_max:)
      @minimum_shape = minimum_shape
      @size_max = size_max
    end

    def call
      exact_shape = even_final_axis(minimum_shape)
      fast_shape = exact_shape.map.with_index do |size, axis|
        next_fast_size(size, even: axis == exact_shape.length - 1)
      end
      [exact_shape, fast_shape].min_by { |shape| transform_cost(shape) }.freeze
    end

    private

    attr_reader :minimum_shape, :size_max

    def even_final_axis(shape)
      result = shape.dup
      result[-1] = checked_add(result[-1], 1) if result[-1].odd?
      result.freeze
    end

    def next_fast_size(target, even:)
      best = nil
      each_power(5) do |power_of_five|
        each_power(3, power_of_five) do |power_of_three|
          candidate = doubled_candidate(power_of_three, target, even)
          best = candidate if candidate && (!best || candidate < best)
        end
      end
      best || raise_overflow
    end

    def each_power(factor, initial = 1)
      value = initial
      loop do
        yield value
        break if value > size_max / factor

        value *= factor
      end
    end

    def doubled_candidate(initial, target, even)
      candidate = initial
      while candidate < target || (even && candidate.odd?)
        return nil if candidate > size_max / 2

        candidate *= 2
      end
      candidate
    end

    def transform_cost(shape)
      checked_product(shape) * shape.sum { |size| Math.log2(size) + large_factor_penalty(size) }
    end

    def large_factor_penalty(size)
      prime_factors(size).sum do |factor|
        factor > 7 ? 1.5 * (Math.log2(factor) - 3) : 0.0
      end
    end

    def prime_factors(value)
      factors = []
      divisor = 2
      while divisor * divisor <= value
        value = extract_factor(value, divisor, factors)
        divisor += divisor == 2 ? 1 : 2
      end
      factors << value if value > 1
      factors
    end

    def extract_factor(value, divisor, factors)
      while (value % divisor).zero?
        factors << divisor
        value /= divisor
      end
      value
    end

    def checked_product(shape)
      shape.reduce(1) do |product, size|
        raise_overflow if !product.zero? && size > size_max / product

        product * size
      end
    end

    def checked_add(left, right)
      return left + right if right <= size_max - left

      raise_overflow
    end

    def raise_overflow
      raise RangeError, 'FFT shape exceeds native implementation limit'
    end
  end

  private_constant :RealFftShape
end
