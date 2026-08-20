# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe '.convolve_basic' do
    it 'works like the example in the README' do
      a = NArray[0.3, 0.4, 0.5]
      b = NArray[1.3, -0.5]
      c = described_class.convolve_basic(a, b)
      expect(c).to be_narray_like NArray[0.19, 0.27]
    end

    it 'calculates a 2D convolution' do
      a = NArray[[0.3, 0.4, 0.5], [0.6, 0.8, 0.2], [0.9, 1.0, 0.1]]
      b = NArray[[1.2, -0.5], [0.5, -1.3]]
      c = described_class.convolve_basic(a, b)
      expect(c).to be_narray_like NArray[[-0.58, 0.37], [-0.53, 1.23]]
    end

    it 'calculates a 2D convolution with rectangular arrays' do
      result = described_class.convolve_basic(
        ConvolutionFixtures.rectangular_signal, ConvolutionFixtures.rectangular_kernel
      )
      expect(result).to be_narray_like ConvolutionFixtures.rectangular_result
    end

    it 'calculates a convolution from non-contiguous views' do
      signal = NArray[[0.3, 0.6, 0.9], [0.4, 0.8, 1.0], [0.5, 0.2, 0.1]].transpose
      kernel = NArray[[1.2, 0.5], [-0.5, -1.3]].transpose

      result = described_class.convolve_basic(signal, kernel)

      expect(result).to be_narray_like NArray[[-0.58, 0.37], [-0.53, 1.23]]
    end

    it 'calculates a 3D convolution' do
      result = described_class.convolve_basic(ConvolutionFixtures.three_dimensional_signal,
                                              ConvolutionFixtures.three_dimensional_kernel)
      expect(result).to be_narray_like ConvolutionFixtures.three_dimensional_result
    end

    it 'calculates a 4D convolution' do
      result = described_class.convolve_basic(ConvolutionFixtures.four_dimensional_signal,
                                              ConvolutionFixtures.four_dimensional_kernel)
      expect(result).to be_narray_like ConvolutionFixtures.four_dimensional_result
    end

    it 'rejects arrays with different ranks' do
      expect { described_class.convolve_basic(NArray.sfloat(2), NArray.sfloat(2, 2)) }
        .to raise_error(ArgumentError, 'signal and kernel must have equal rank')
    end

    it 'rejects a signal that is not a Numo array' do
      expect { described_class.convolve_basic([1.0], NArray[1.0]) }
        .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
    end

    it 'rejects a kernel that is not a Numo array' do
      expect { described_class.convolve_basic(NArray[1.0], [1.0]) }
        .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
    end

    it 'rejects an empty signal' do
      expect { described_class.convolve_basic(NArray.zeros(0), NArray[1.0]) }
        .to raise_error(ArgumentError, 'signal and kernel must not be empty')
    end

    it 'rejects an empty kernel' do
      expect { described_class.convolve_basic(NArray[1.0], NArray.zeros(0)) }
        .to raise_error(ArgumentError, 'signal and kernel must not be empty')
    end

    it 'rejects arrays above the maximum supported rank' do
      rank_17_shape = Array.new(17, 1)
      rank_17_array = NArray.sfloat(*rank_17_shape)

      expect { described_class.convolve_basic(rank_17_array, rank_17_array) }
        .to raise_error(ArgumentError, 'maximum supported rank is 16')
    end

    it 'calculates a convolution at the maximum supported rank' do
      rank_16_shape = Array.new(16, 1)
      signal = NArray.ones(*rank_16_shape) * 2
      kernel = NArray.ones(*rank_16_shape) * 3

      result = described_class.convolve_basic(signal, kernel)

      expect(result).to be_narray_like NArray.ones(*rank_16_shape) * 6
    end

    it 'rejects a kernel larger than the signal' do
      expect { described_class.convolve_basic(NArray.sfloat(2), NArray.sfloat(3)) }
        .to raise_error(ArgumentError, 'kernel must not be larger than signal in any dimension')
    end
  end
end
