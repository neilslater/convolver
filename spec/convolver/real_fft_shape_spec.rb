# frozen_string_literal: true

require 'helpers'

module Convolver
  describe RealFftShape do
    it 'rejects a mandatory transform whose element count exceeds the native limit' do
      expect { described_class.new([8, 10], size_max: 72).call }
        .to raise_error(RangeError, 'FFT shape exceeds native implementation limit')
    end

    it 'rejects an odd final axis when its mandatory even increment exceeds the native limit' do
      expect { described_class.new([3], size_max: 3).call }
        .to raise_error(RangeError, 'FFT shape exceeds native implementation limit')
    end
  end
end
