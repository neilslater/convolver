# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe '.convolve' do
    it 'works like the example in the README' do
      a = Numo::SFloat[0.3, 0.4, 0.5]
      b = Numo::SFloat[1.3, -0.5]
      c = described_class.convolve(a, b)
      expect(c).to be_narray_like Numo::SFloat[0.19, 0.27]
    end

    it 'processes convolutions of different sizes' do
      # The variety here is to ensure all branches of optimisation algorithm
      # are covered
      [10, 30, 60, 90, 100, 120, 130, 150, 175, 200].each do |asize|
        [5, 10, 12, 15, 20, 30, 40, 50].each do |bsize|
          next unless bsize < asize

          a = Numo::SFloat.new(asize, asize).rand
          b = Numo::SFloat.new(bsize, bsize).rand
          c = described_class.convolve(a, b)

          # We should always match output of convolve_basic irrespective
          # of what the optimal choice of algorithm is (larger error allowed here due to rounding)
          expect_result = described_class.convolve_basic(a, b)
          expect(c).to be_narray_like(expect_result, 1e-6)
        end
      end
    end

    it 'chooses #convolve_basic for small inputs' do
      a = Numo::SFloat.new(50, 50).rand
      b = Numo::SFloat.new(10, 10).rand
      allow(described_class).to receive(:convolve_basic)
      allow(described_class).to receive(:convolve_fft)
      described_class.convolve(a, b)
      expect(described_class).to have_received(:convolve_basic).once
      expect(described_class).not_to have_received(:convolve_fft)
    end

    it 'chooses .convolve_fft for large inputs' do
      a = Numo::SFloat.new(500, 500).rand
      b = Numo::SFloat.new(100, 100).rand
      allow(described_class).to receive(:convolve_fft)
      allow(described_class).to receive(:convolve_basic)
      described_class.convolve(a, b)
      expect(described_class).to have_received(:convolve_fft).once
      expect(described_class).not_to have_received(:convolve_basic)
    end

    it 'rejects values that are not Numo arrays' do
      expect { described_class.convolve([1.0], Numo::SFloat[1.0]) }
        .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
    end

    it 'rejects arrays with different ranks' do
      expect { described_class.convolve(Numo::SFloat.zeros(2), Numo::SFloat.zeros(2, 2)) }
        .to raise_error(ArgumentError, 'signal and kernel must have equal rank')
    end

    it 'rejects a kernel larger than the signal' do
      expect { described_class.convolve(Numo::SFloat.zeros(2), Numo::SFloat.zeros(3)) }
        .to raise_error(ArgumentError, 'kernel must not be larger than signal in any dimension')
    end
  end

  describe '.convolve_fftw3' do
    it 'warns and delegates all options to .convolve_fft' do
      signal = Numo::SFloat[0.3, 0.4, 0.5]
      kernel = Numo::SFloat[1.3, -0.5]
      allow(described_class).to receive(:convolve_fft).and_call_original

      expect { described_class.convolve_fftw3(signal, kernel, mode: :full, fill_value: -1) }
        .to output(/deprecated; use \.convolve_fft/).to_stderr
      expect(described_class).to have_received(:convolve_fft)
        .with(signal, kernel, mode: :full, boundary: :constant, fill_value: -1, origin: 0)
    end
  end
end
