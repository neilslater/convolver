# frozen_string_literal: true

require 'convolver/buffer_limits'

# Internal allocation and arithmetic checks for the PocketFFT paths.
module Convolver
  # Only FFT representability failures permit automatic direct fallback.
  # @private
  class FftUnavailable < RangeError; end

  # Bounds PocketFFT buffers and signed arithmetic without constructing plans.
  # @private
  class FftLimits
    def initialize(buffers = BufferLimits::NATIVE)
      @buffers = buffers
    end

    def validate!(shape, real_axis:)
      total = buffers.product!(shape, 'FFT size', error: FftUnavailable)
      validate_buffers!(shape, total, real_axis)
      # The wrapper casts total to int before dividing it by the axis length.
      integer!(total, buffers.int_max, 'FFT total size')
      # Real forward computes int z_step = (z / 2 + 1) * 2.
      integer!(shape[real_axis], buffers.int_max - 2, 'FFT real stride') if real_axis
      validate_axes!(shape)
    end

    private

    attr_reader :buffers

    def validate_axes!(shape)
      # FFTPACK twiddle counts are <= 2n complex / 3n real. Bluestein m < 4n,
      # so each variable allocation is <= 128n bytes, including inner twiddles.
      # The 16n int bound also protects signed intermediates at the inner m.
      shape.each do |length|
        integer!(length, buffers.int_max / 16, 'FFT axis arithmetic')
        buffer!(length, 128, 'FFT axis work buffer')
      end
    end

    def validate_buffers!(shape, total, real_axis)
      buffer!(total, 8, 'FFT spatial buffer')
      return buffer!(total, 16, 'FFT complex buffer') unless real_axis

      spectrum = total / shape[real_axis] * ((shape[real_axis] / 2) + 1)
      buffer!(spectrum, 16, 'FFT spectrum buffer')
      buffer!(total, 16, 'FFT inverse staging buffer')
      buffer!(total, 8, 'FFT inverse output buffer')
    end

    def buffer!(count, width, description)
      buffers.buffer!(count, width, description, error: FftUnavailable)
    end

    def integer!(value, maximum, description)
      return if value <= maximum

      raise FftUnavailable, "#{description} exceeds native integer limit"
    end
  end

  private_constant :FftUnavailable, :FftLimits
end
