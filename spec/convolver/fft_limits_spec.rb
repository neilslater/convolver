# frozen_string_literal: true

require 'helpers'

module Convolver
  describe FftLimits do
    def check(shape, byte_max: BufferLimits::BYTE_MAX, real_axis: shape.length - 1)
      described_class.new(BufferLimits.new(byte_max:)).validate!(shape, real_axis:)
    end

    it 'rejects an overflowing transform element product' do
      expect { check([BufferLimits::SIZE_MAX, 2]) }
        .to raise_error(FftUnavailable, /FFT size exceeds native element limit/)
    end

    it 'checks spatial byte multiplication' do
      expect { check([7, 10], byte_max: 559) }
        .to raise_error(FftUnavailable, /FFT spatial buffer/)
    end

    it 'checks the real half-spectrum independently' do
      expect { check([7, 10], byte_max: 600) }
        .to raise_error(FftUnavailable, /FFT spectrum buffer/)
    end

    it 'checks full complex staging even when the half-spectrum fits' do
      expect { check([7, 10], byte_max: 800) }
        .to raise_error(FftUnavailable, /FFT inverse staging buffer/)
    end

    it 'checks full complex buffers for all-odd periodic transforms' do
      expect { check([3, 5], byte_max: 239, real_axis: nil) }
        .to raise_error(FftUnavailable, /FFT complex buffer/)
    end

    it 'checks axis work even when all transform-sized buffers fit' do
      expect { check([7, 10], byte_max: 1120) }
        .to raise_error(FftUnavailable, /FFT axis work buffer/)
    end

    it 'accepts the conservative work-buffer boundary' do
      expect { check([1, 16], byte_max: 2048) }.not_to raise_error
    end

    it 'rejects the next axis beyond the work-buffer boundary' do
      expect { check([1, 18], byte_max: 2048) }
        .to raise_error(FftUnavailable, /FFT axis work buffer/)
    end

    it 'rejects the cast-before-division overflow fixture' do
      expect { check([65_536, 65_536]) }
        .to raise_error(FftUnavailable, /FFT total size exceeds native integer limit/)
    end

    it 'rejects the real z_step signed multiplication fixture' do
      expect { check([2_147_483_646]) }
        .to raise_error(FftUnavailable, /FFT real stride exceeds native integer limit/)
    end

    it 'accepts the conservative integer axis boundary' do
      expect { check([134_217_727, 2]) }.not_to raise_error
    end

    it 'rejects the next axis to protect planner signed intermediates' do
      expect { check([134_217_728, 2]) }
        .to raise_error(FftUnavailable, /FFT axis arithmetic exceeds native integer limit/)
    end
  end
end
