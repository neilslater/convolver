# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe '#convolve' do
    it 'works like the example in the README' do
      a = NArray[0.3, 0.4, 0.5]
      b = NArray[1.3, -0.5]
      c = described_class.convolve(a, b)
      expect(c).to be_narray_like NArray[0.19, 0.27]
    end

    it 'processes convolutions of different sizes' do
      # The variety here is to ensure all branches of optimisation algorithm
      # are covered
      [10, 30, 60, 90, 100, 120, 130, 150, 175, 200].each do |asize|
        [5, 10, 12, 15, 20, 30, 40, 50].each do |bsize|
          next unless bsize < asize

          a = NArray.sfloat(asize, asize).random
          b = NArray.sfloat(bsize, bsize).random
          c = described_class.convolve(a, b)

          # We should always match output of convolve_basic irrespective
          # of what the optimal choice of algorithm is (larger error allowed here due to rounding)
          expect_result = described_class.convolve_basic(a, b)
          expect(c).to be_narray_like(expect_result, 1e-6)
        end
      end
    end

    it 'chooses #convolve_basic for small inputs' do
      a = NArray.sfloat(50, 50).random
      b = NArray.sfloat(10, 10).random
      allow(described_class).to receive(:convolve_basic)
      allow(described_class).to receive(:convolve_fftw3)
      described_class.convolve(a, b)
      expect(described_class).to have_received(:convolve_basic).once
      expect(described_class).not_to have_received(:convolve_fftw3)
    end

    it 'chooses #convolve_fftw3 for large inputs' do
      a = NArray.sfloat(500, 500).random
      b = NArray.sfloat(100, 100).random
      allow(described_class).to receive(:convolve_fftw3)
      allow(described_class).to receive(:convolve_basic)
      described_class.convolve(a, b)
      expect(described_class).to have_received(:convolve_fftw3).once
      expect(described_class).not_to have_received(:convolve_basic)
    end
  end
end
