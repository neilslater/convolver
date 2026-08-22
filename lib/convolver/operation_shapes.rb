# frozen_string_literal: true

require 'convolver/real_fft_shape'

module Convolver
  # Calculates and validates output, extension, and FFT dimensions.
  class OperationShapes
    SIZE_MAX = (1 << ([0].pack('J').bytesize * 8)) - 1

    attr_reader :padding_before, :padding_after, :result_shape,
                :extended_shape, :result_size, :extended_size

    def initialize(signal_shape, kernel_shape, operation:, mode:, anchors:)
      @operation = operation
      @mode = mode
      calculate_shapes(signal_shape, kernel_shape, anchors)
      validate_sizes!
    end

    def linear_fft_shape(kernel_shape)
      return [].freeze if kernel_shape.empty?

      RealFftShape.new(linear_result_shape(kernel_shape), size_max: SIZE_MAX).call
    end

    def linear_fft_size(kernel_shape)
      checked_product(linear_fft_shape(kernel_shape), 'FFT size')
    end

    def linear_spectrum_size(kernel_shape)
      shape = linear_fft_shape(kernel_shape)
      return 1 if shape.empty?

      spectrum_shape = shape.dup
      spectrum_shape[-1] = (spectrum_shape[-1] / 2) + 1
      checked_product(spectrum_shape, 'FFT spectrum size')
    end

    private

    attr_reader :operation, :mode

    def calculate_shapes(signal_shape, kernel_shape, anchors)
      @padding_before, @padding_after, @result_shape = shapes_for(signal_shape, kernel_shape, anchors)
      @extended_shape = signal_shape.zip(padding_before, padding_after).map do |signal_size, before, after|
        signal_size + before + after
      end
    end

    def shapes_for(signal_shape, kernel_shape, anchors)
      case mode
      when :valid then valid_shapes(signal_shape, kernel_shape)
      when :same then same_shapes(signal_shape, kernel_shape, anchors)
      when :full then full_shapes(signal_shape, kernel_shape)
      end
    end

    def valid_shapes(signal_shape, kernel_shape)
      result = signal_shape.zip(kernel_shape).map do |signal_size, kernel_size|
        signal_size - kernel_size + 1
      end
      zeros = Array.new(signal_shape.length, 0).freeze
      [zeros, zeros, result]
    end

    def same_shapes(signal_shape, kernel_shape, anchors)
      correlation_before = anchors.dup
      correlation_after = kernel_shape.zip(anchors).map { |kernel_size, anchor| kernel_size - 1 - anchor }
      before, after = if operation == :correlation
                        [correlation_before, correlation_after]
                      else
                        [correlation_after, correlation_before]
                      end
      before.freeze
      after.freeze
      [before, after, signal_shape.dup]
    end

    def full_shapes(signal_shape, kernel_shape)
      padding = kernel_shape.map { |kernel_size| kernel_size - 1 }.freeze
      result = signal_shape.zip(kernel_shape).map { |signal_size, kernel_size| signal_size + kernel_size - 1 }
      [padding, padding, result]
    end

    def validate_sizes!
      @padding_before, @padding_after, @result_shape, @extended_shape = [
        [padding_before, 'padding shape'],
        [padding_after, 'padding shape'],
        [result_shape, 'result shape'],
        [extended_shape, 'extended signal shape']
      ].map { |shape, description| checked_shape(shape, description).freeze }
      @result_size = checked_product(result_shape, 'result size')
      @extended_size = checked_product(extended_shape, 'extended signal size')
    end

    def checked_shape(shape, description)
      return shape if shape.all? { |size| size.between?(0, SIZE_MAX) }

      raise RangeError, "#{description} exceeds native implementation limit"
    end

    def checked_product(shape, description)
      shape.reduce(1) do |product, size|
        if !product.zero? && size > SIZE_MAX / product
          raise RangeError, "#{description} exceeds native implementation limit"
        end

        product * size
      end
    end

    def linear_result_shape(kernel_shape)
      shape = extended_shape.zip(kernel_shape).map do |signal_size, kernel_size|
        checked_add(signal_size, kernel_size - 1, 'FFT shape')
      end
      checked_shape(shape, 'FFT shape')
    end

    def checked_add(left, right, description)
      return left + right if right <= SIZE_MAX - left

      raise RangeError, "#{description} exceeds native implementation limit"
    end
  end

  private_constant :OperationShapes
end
