# frozen_string_literal: true

module Convolver
  # Validates and normalizes the options for a correlation operation.
  class OperationOptions
    MODES = %i[valid same full].freeze
    BOUNDARIES = %i[constant nearest reflect mirror wrap].freeze

    attr_reader :mode, :boundary, :fill_value, :origins, :anchors

    def initialize(signal, kernel, mode:, boundary:, fill_value:, origin:)
      validate_inputs!(signal, kernel)
      validate_vocabulary!(mode, boundary)
      assign_options(signal, kernel, mode:, boundary:, fill_value:, origin:)
      validate_combinations!(signal.shape, kernel.shape)
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

    def assign_options(signal, kernel, mode:, boundary:, fill_value:, origin:)
      @mode = mode
      @boundary = boundary
      @fill_value_given = !fill_value.equal?(UNSPECIFIED_FILL)
      @fill_value = normalize_fill_value(fill_value)
      @origins = normalize_origins(origin, signal.ndim)
      @anchors = kernel.shape.zip(@origins).map.with_index do |(kernel_size, axis_origin), axis|
        normalize_anchor(kernel_size, axis_origin, axis)
      end.freeze
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
  end

  private_constant :OperationOptions
end
