# frozen_string_literal: true

# Independent finite-array definitions for convolution and correlation specs.
module OperationReference
  OPERATIONS = %i[correlation convolution].freeze
  CALCULATION_METHODS = {
    correlation: %i[correlate correlate_basic correlate_fft],
    convolution: %i[convolve convolve_basic convolve_fft]
  }.freeze
  ESTIMATOR_METHODS = {
    correlation: %i[predict_correlate_basic_time predict_correlate_fft_time],
    convolution: %i[predict_convolve_basic_time predict_convolve_fft_time]
  }.freeze
  ENTRY_POINTS = (CALCULATION_METHODS.values.flatten + ESTIMATOR_METHODS.values.flatten).freeze

  # Evaluates one operation independently of Convolver's planning code.
  class Calculation
    def initialize(operation, signal, kernel, mode:, boundary:, fill_value:, origin:)
      @operation = operation
      @signal = signal
      @kernel = kernel
      @boundary = boundary
      @fill_value = fill_value
      origins = origin.is_a?(Array) ? origin : Array.new(signal.ndim, origin)
      anchors = kernel.shape.zip(origins).map { |size, shift| (size / 2) + shift }
      @result_shape, @starts = OperationReference.result_layout(
        operation, signal.shape, kernel.shape, anchors, mode
      )
    end

    def call
      result = Numo::SFloat.zeros(*result_shape)
      OperationReference.coordinates(result_shape).each do |output_coordinate|
        result[*output_coordinate] = result_at(output_coordinate)
      end
      result
    end

    private

    attr_reader :operation, :signal, :kernel, :boundary, :fill_value, :result_shape, :starts

    def result_at(output_coordinate)
      OperationReference.coordinates(kernel.shape).sum do |kernel_coordinate|
        signal_coordinate = signal_coordinate(output_coordinate, kernel_coordinate)
        value = OperationReference.boundary_value(signal, signal_coordinate, boundary, fill_value)
        value * kernel[*kernel_coordinate]
      end
    end

    def signal_coordinate(output_coordinate, kernel_coordinate)
      output_coordinate.zip(kernel_coordinate, starts).map do |output, kernel_index, start|
        output + start + (kernel_direction * kernel_index)
      end
    end

    def kernel_direction
      operation == :correlation ? 1 : -1
    end
  end

  module_function

  def calculate(operation, signal, kernel, mode:, boundary: :constant, fill_value: 0.0, origin: 0)
    Calculation.new(operation, signal, kernel, mode:, boundary:, fill_value:, origin:).call
  end

  def coordinates(shape)
    shape.reduce([[]]) do |coordinates, size|
      coordinates.flat_map { |prefix| size.times.map { |index| prefix + [index] } }
    end
  end

  def result_layout(operation, signal_shape, kernel_shape, anchors, mode)
    result_shape = result_shape(signal_shape, kernel_shape, mode)
    starts = starts(operation, kernel_shape, anchors, mode)
    [result_shape, starts]
  end

  def result_shape(signal_shape, kernel_shape, mode)
    case mode
    when :valid then signal_shape.zip(kernel_shape).map { |signal, kernel| signal - kernel + 1 }
    when :same then signal_shape
    when :full then signal_shape.zip(kernel_shape).map { |signal, kernel| signal + kernel - 1 }
    end
  end

  def starts(operation, kernel_shape, anchors, mode)
    return correlation_starts(kernel_shape, anchors, mode) if operation == :correlation

    convolution_starts(kernel_shape, anchors, mode)
  end

  def correlation_starts(kernel_shape, anchors, mode)
    case mode
    when :valid then Array.new(kernel_shape.length, 0)
    when :same then anchors.map(&:-@)
    when :full then kernel_shape.map { |kernel| 1 - kernel }
    end
  end

  def convolution_starts(kernel_shape, anchors, mode)
    case mode
    when :valid then kernel_shape.map { |kernel| kernel - 1 }
    when :same then anchors
    when :full then Array.new(kernel_shape.length, 0)
    end
  end

  def boundary_value(signal, coordinate, boundary, fill_value)
    return signal[*coordinate] if coordinate.zip(signal.shape).all? { |index, size| index.between?(0, size - 1) }
    return fill_value if boundary == :constant

    mapped = coordinate.zip(signal.shape).map { |index, size| boundary_index(index, size, boundary) }
    signal[*mapped]
  end

  def boundary_index(index, size, boundary)
    case boundary
    when :nearest then index.clamp(0, size - 1)
    when :wrap then index % size
    when :reflect then reflect_index(index, size)
    when :mirror then mirror_index(index, size)
    end
  end

  def reflect_index(index, size)
    residue = index % (2 * size)
    residue < size ? residue : (2 * size) - 1 - residue
  end

  def mirror_index(index, size)
    return 0 if size == 1

    period = 2 * (size - 1)
    residue = index % period
    residue < size ? residue : period - residue
  end
end
