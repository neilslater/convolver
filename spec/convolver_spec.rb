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
      expect_optimized_convolutions_to_match_basic
    end

    context 'with small inputs' do
      before { exercise_algorithm_selection(50, 10) }

      it 'chooses .convolve_basic' do
        expect(described_class).to have_received(:convolve_basic).once
      end

      it 'does not choose .convolve_fft' do
        expect(described_class).not_to have_received(:convolve_fft)
      end
    end

    context 'with large inputs' do
      before { exercise_algorithm_selection(500, 100) }

      it 'chooses .convolve_fft' do
        expect(described_class).to have_received(:convolve_fft).once
      end

      it 'does not choose .convolve_basic' do
        expect(described_class).not_to have_received(:convolve_basic)
      end
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

    def expect_optimized_convolutions_to_match_basic
      signal_sizes = [10, 30, 60, 90, 100, 120, 130, 150, 175, 200]
      kernel_sizes = [5, 10, 12, 15, 20, 30, 40, 50]
      signal_sizes.product(kernel_sizes).select { |signal_size, kernel_size| kernel_size < signal_size }.each do |sizes|
        expect_optimized_size_pair_to_match_basic(sizes)
      end
    end

    def expect_optimized_size_pair_to_match_basic(sizes)
      signal = Numo::SFloat.new(sizes.first, sizes.first).rand
      kernel = Numo::SFloat.new(sizes.last, sizes.last).rand
      expected = described_class.convolve_basic(signal, kernel)
      expect(described_class.convolve(signal, kernel)).to be_narray_like(expected, 1e-6)
    end

    def exercise_algorithm_selection(signal_size, kernel_size)
      signal = Numo::SFloat.new(signal_size, signal_size).rand
      kernel = Numo::SFloat.new(kernel_size, kernel_size).rand
      allow(described_class).to receive(:convolve_basic)
      allow(described_class).to receive(:convolve_fft)
      described_class.convolve(signal, kernel)
    end
  end

  describe '.convolve_fftw3' do
    let(:signal) { Numo::SFloat[0.3, 0.4, 0.5] }
    let(:kernel) { Numo::SFloat[1.3, -0.5] }

    it 'warns about the deprecated name' do
      expect { invoke_deprecated_convolution }
        .to output(/deprecated; use \.convolve_fft/).to_stderr
    end

    it 'delegates all options to .convolve_fft' do
      allow(described_class).to receive(:warn)
      allow(described_class).to receive(:convolve_fft).and_call_original
      invoke_deprecated_convolution
      expect(described_class).to have_received(:convolve_fft)
        .with(signal, kernel, mode: :full, boundary: :constant, fill_value: -1, origin: 0)
    end

    def invoke_deprecated_convolution
      described_class.convolve_fftw3(signal, kernel, mode: :full, fill_value: -1)
    end
  end
end
