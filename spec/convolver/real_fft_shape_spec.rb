# frozen_string_literal: true

require 'helpers'

module Convolver
  describe RealFftShape do
    it 'retains the exact shape when the fast candidate product overflows' do
      expect(described_class.new([7, 10], size_max: 72).call).to eq [7, 10]
    end

    it 'retains an even exact shape when no next fast length is representable' do
      expect(described_class.new([14], size_max: 14).call).to eq [14]
    end

    it 'retains the mandatory even increment when the next fast length overflows' do
      expect(described_class.new([13], size_max: 14).call).to eq [14]
    end

    it 'accepts an exact element count at the limit' do
      expect(described_class.new([7, 10], size_max: 70).call).to eq [7, 10]
    end

    it 'still selects a cheaper fast shape when both candidates fit' do
      expect(described_class.new([22], size_max: 24).call).to eq [24]
    end

    it 'returns a frozen shape' do
      expect(described_class.new([7, 9], size_max: 72).call).to be_frozen
    end

    it 'does not change the supplied minimum' do
      minimum = [7, 9]
      described_class.new(minimum, size_max: 72).call

      expect(minimum).to eq [7, 9]
    end

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
