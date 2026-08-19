# frozen_string_literal: true

# Internal planning and extension support for Convolver's public operations.
module Convolver
  # Distinguishes an omitted fill_value keyword from an explicitly supplied
  # value. This lets non-constant boundaries reject even an explicit zero.
  UNSPECIFIED_FILL = Object.new.freeze

  # Validated dimensions and boundary-extension details for one correlation.
  class OperationPlan # rubocop:disable Metrics/ClassLength
    MODES = %i[valid same full].freeze
    BOUNDARIES = %i[constant nearest reflect mirror wrap].freeze
    SIZE_MAX = (1 << ([0].pack('J').bytesize * 8)) - 1

    attr_reader :mode, :boundary, :fill_value, :origins, :anchors,
                :padding_before, :padding_after, :result_shape,
                :extended_shape, :result_size, :extended_size

    # The public operation has two positional and four keyword parameters.
    # rubocop:disable Metrics/ParameterLists
    def initialize(signal, kernel, mode:, boundary:, fill_value:, origin:)
      validate_inputs!(signal, kernel)
      validate_vocabulary!(mode, boundary)
      assign_options(signal, kernel, mode, boundary, fill_value, origin)
      validate_combinations!(signal.shape, kernel.shape)
      calculate_shapes(signal.shape, kernel.shape)
      validate_sizes!
    end

    def assign_options(signal, kernel, mode, boundary, fill_value, origin)
      @mode = mode
      @boundary = boundary
      @fill_value_given = !fill_value.equal?(UNSPECIFIED_FILL)
      @fill_value = normalize_fill_value(fill_value)
      @origins = normalize_origins(origin, signal.ndim)
      @anchors = kernel.shape.zip(@origins).map.with_index do |(kernel_size, axis_origin), axis|
        normalize_anchor(kernel_size, axis_origin, axis)
      end.freeze
    end
    # rubocop:enable Metrics/ParameterLists

    def valid?
      mode == :valid
    end

    def wrap?
      mode == :same && boundary == :wrap
    end

    def linear_fft_shape(kernel_shape)
      checked_shape(
        extended_shape.zip(kernel_shape).map { |signal_size, kernel_size| signal_size + kernel_size - 1 },
        'FFT shape'
      )
    end

    def linear_fft_size(kernel_shape)
      checked_product(linear_fft_shape(kernel_shape), 'FFT size')
    end

    def extend_signal(signal)
      return signal if valid?

      source = Numo::SFloat.cast(signal)
      return constant_extension(source) if boundary == :constant

      source[*extension_indices]
    end

    private

    def validate_inputs!(signal, kernel)
      unless signal.is_a?(Numo::NArray) && kernel.is_a?(Numo::NArray)
        raise ArgumentError, 'signal and kernel must be Numo::NArray values'
      end
      raise ArgumentError, 'signal and kernel must not be empty' if signal.empty? || kernel.empty?
      raise ArgumentError, 'signal and kernel must have equal rank' unless signal.ndim == kernel.ndim
      raise ArgumentError, "maximum supported rank is #{MAX_RANK}" if signal.ndim > MAX_RANK
    end

    def validate_vocabulary!(mode, boundary)
      raise ArgumentError, "mode must be one of #{MODES.inspect}" unless MODES.include?(mode)
      return if BOUNDARIES.include?(boundary)

      raise ArgumentError, "boundary must be one of #{BOUNDARIES.inspect}"
    end

    def normalize_fill_value(fill_value)
      value = fill_value.equal?(UNSPECIFIED_FILL) ? 0.0 : fill_value
      unless value.is_a?(Numeric) && !value.is_a?(Complex)
        raise ArgumentError, 'fill_value must be a real numeric value'
      end

      value.to_f
    end

    def normalize_origins(origin, rank)
      origins = case origin
                when Integer then Array.new(rank, origin)
                when Array then origin.dup
                else
                  raise ArgumentError, 'origin must be an Integer or an Array of one Integer per dimension'
                end

      unless origins.length == rank && origins.all?(Integer)
        raise ArgumentError, 'origin must contain one Integer per dimension'
      end

      origins.freeze
    end

    def normalize_anchor(kernel_size, origin, axis)
      anchor = (kernel_size / 2) + origin
      return anchor if anchor.between?(0, kernel_size - 1)

      raise ArgumentError,
            "origin #{origin} is out of range for kernel dimension #{axis} of size #{kernel_size}"
    end

    def validate_combinations!(signal_shape, kernel_shape)
      validate_mode_combination!(signal_shape, kernel_shape)
      validate_fill_combination!
      validate_origin_combination!
    end

    def validate_mode_combination!(signal_shape, kernel_shape)
      return validate_valid_options!(signal_shape, kernel_shape) if mode == :valid
      return unless mode == :full && boundary != :constant

      raise ArgumentError, 'mode: :full only supports boundary: :constant'
    end

    def validate_fill_combination!
      return unless boundary != :constant && @fill_value_given

      raise ArgumentError, 'fill_value is only supported with boundary: :constant'
    end

    def validate_origin_combination!
      return if mode == :same || origins.all?(&:zero?)

      raise ArgumentError, 'nonzero origin is only supported with mode: :same'
    end

    def validate_valid_options!(signal_shape, kernel_shape)
      unless signal_shape.zip(kernel_shape).all? { |signal_size, kernel_size| signal_size >= kernel_size }
        raise ArgumentError, 'kernel must not be larger than signal in any dimension'
      end
      raise ArgumentError, 'mode: :valid only supports boundary: :constant' unless boundary == :constant
      raise ArgumentError, 'mode: :valid requires fill_value: 0' unless fill_value.zero?
    end

    def calculate_shapes(signal_shape, kernel_shape)
      @padding_before, @padding_after, @result_shape = case mode
                                                       when :valid then valid_shapes(signal_shape, kernel_shape)
                                                       when :same then same_shapes(signal_shape, kernel_shape)
                                                       when :full then full_shapes(signal_shape, kernel_shape)
                                                       end
      @extended_shape = signal_shape.zip(padding_before, padding_after).map do |signal_size, before, after|
        signal_size + before + after
      end
    end

    def valid_shapes(signal_shape, kernel_shape)
      result = signal_shape.zip(kernel_shape).map do |signal_size, kernel_size|
        signal_size - kernel_size + 1
      end
      zeros = Array.new(signal_shape.length, 0).freeze
      [zeros, zeros, result]
    end

    def same_shapes(signal_shape, kernel_shape)
      before = anchors.dup.freeze
      after = kernel_shape.zip(anchors).map { |kernel_size, anchor| kernel_size - 1 - anchor }.freeze
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

    def constant_extension(source)
      extended = Numo::SFloat.new(*extended_shape).fill(fill_value)
      ranges = source.shape.zip(padding_before).map do |signal_size, before|
        before...(before + signal_size)
      end
      extended[*ranges] = source
      extended
    end

    def extension_indices
      extended_shape.zip(padding_before).map.with_index do |(length, before), axis|
        signal_size = extended_shape[axis] - before - padding_after[axis]
        Array.new(length) { |offset| boundary_index(offset - before, signal_size) }
      end
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

  private_constant :OperationPlan, :UNSPECIFIED_FILL
end
