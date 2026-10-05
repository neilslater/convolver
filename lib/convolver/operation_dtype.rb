# frozen_string_literal: true

module Convolver
  # Resolves public precision without allocating operand copies.
  # @private
  class OperationDtype
    SINGLE_INPUTS = [Numo::SFloat, Numo::Int8, Numo::UInt8, Numo::Int16, Numo::UInt16].freeze
    DOUBLE_INPUTS = [Numo::DFloat, Numo::Int32, Numo::UInt32, Numo::Int64, Numo::UInt64].freeze
    INPUTS = (SINGLE_INPUTS + DOUBLE_INPUTS).freeze
    OUTPUTS = [Numo::SFloat, Numo::DFloat].freeze

    def self.resolve(signal, kernel, dtype)
      validate_inputs!(signal, kernel)
      validate_dtype!(dtype)
      types = [signal.class, kernel.class]
      dtype || (types.any? { |type| DOUBLE_INPUTS.include?(type) } ? Numo::DFloat : Numo::SFloat)
    end

    def self.validate_inputs!(signal, kernel)
      unless signal.is_a?(Numo::NArray) && kernel.is_a?(Numo::NArray)
        raise ArgumentError, 'signal and kernel must be Numo::NArray values'
      end

      return if [signal.class, kernel.class].all? { |type| INPUTS.include?(type) }

      raise ArgumentError, 'unsupported signal or kernel dtype'
    end

    def self.validate_dtype!(dtype)
      return if dtype.nil? || OUTPUTS.include?(dtype)

      raise ArgumentError, 'dtype must be Numo::SFloat, Numo::DFloat, or nil'
    end

    def self.cast_fill(value, dtype)
      # A scalar IEEE float conversion, without allocating a Numo buffer before preflight.
      dtype == Numo::SFloat ? [value].pack('f').unpack1('f') : value
    end
  end

  private_constant :OperationDtype
end
