# frozen_string_literal: true

module Convolver
  # Checks individual allocations against native size and tagged-stride limits.
  # @private
  class BufferLimits
    SIZE_MAX = (1 << ([0].pack('J').bytesize * 8)) - 1
    # Numo tags signed strides with a left shift: floor(SSIZE_MAX / 2).
    BYTE_MAX = SIZE_MAX / 4
    INT_MAX = (1 << (([0].pack('i').bytesize * 8) - 1)) - 1
    INDEX_BYTES = [0].pack('J').bytesize

    attr_reader :size_max, :byte_max, :int_max

    def initialize(size_max: SIZE_MAX, byte_max: BYTE_MAX, int_max: INT_MAX)
      @size_max = size_max
      @byte_max = byte_max
      @int_max = int_max
    end

    def buffer!(count, width, description, error: RangeError)
      return if count <= byte_max / width

      raise error, "#{description} exceeds native byte/stride limit"
    end

    def product!(shape, description, error: RangeError)
      shape.reduce(1) do |product, size|
        raise error, "#{description} exceeds native element limit" if size > size_max / product

        product * size
      end
    end

    NATIVE = new.freeze
  end

  private_constant :BufferLimits
end
