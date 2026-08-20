# frozen_string_literal: true

module Convolver
  # Extends a signal according to one operation's boundary policy.
  class SignalExtension
    def initialize(shapes, boundary:, fill_value:)
      @shapes = shapes
      @boundary = boundary
      @fill_value = fill_value
    end

    def call(signal)
      source = Numo::SFloat.cast(signal)
      return constant_extension(source) if boundary == :constant

      source[*extension_indices]
    end

    private

    attr_reader :shapes, :boundary, :fill_value

    def constant_extension(source)
      extended = Numo::SFloat.new(*shapes.extended_shape).fill(fill_value)
      ranges = source.shape.zip(shapes.padding_before).map do |signal_size, before|
        before...(before + signal_size)
      end
      extended[*ranges] = source
      extended
    end

    def extension_indices
      shapes.extended_shape.zip(shapes.padding_before).map.with_index do |(length, before), axis|
        Array.new(length) { |offset| boundary_index(offset - before, signal_size(axis, before)) }
      end
    end

    def signal_size(axis, before)
      shapes.extended_shape[axis] - before - shapes.padding_after[axis]
    end

    def boundary_index(position, size)
      case boundary
      when :nearest then position.clamp(0, size - 1)
      when :wrap then position % size
      when :reflect then reflect_index(position, size)
      when :mirror then mirror_index(position, size)
      end
    end

    def reflect_index(position, size)
      residue = position % (2 * size)
      residue < size ? residue : (2 * size) - 1 - residue
    end

    def mirror_index(position, size)
      return 0 if size == 1

      period = 2 * (size - 1)
      residue = position % period
      residue < size ? residue : period - residue
    end
  end

  private_constant :SignalExtension
end
