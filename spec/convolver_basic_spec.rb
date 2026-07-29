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
      a = NArray[ [0.3, 0.4, 0.5, 0.3, 0.4], [0.6, 0.8, 0.2, 0.8, 0.2],
                  [0.9, 1.0, 0.1, 0.9, 1.0], [0.5, 0.9, 0.3, 0.2, 0.8], [0.7, 0.1, 0.3, 0.0, 0.1],
                  [0.4, 0.5, 0.6, 0.7, 0.8], [0.5, 0.4, 0.3, 0.2, 0.1] ]
      b = NArray[[1.2, -0.5, 0.2], [1.8, 0.5, -1.3]]
      c = described_class.convolve_basic(a, b)
      expect(c).to be_narray_like NArray[ [1.48, 0.79, 1.03], [2.35, 1.7, -0.79], [1.56, 2.84, -0.53],
                                          [1.13, 1.3, 0.83], [1.04, 0.26, 0.77], [1.06, 1.05, 1.04] ]
    end

    it 'calculates a convolution from non-contiguous views' do
      signal = NArray[[0.3, 0.6, 0.9], [0.4, 0.8, 1.0], [0.5, 0.2, 0.1]].transpose
      kernel = NArray[[1.2, 0.5], [-0.5, -1.3]].transpose

      result = described_class.convolve_basic(signal, kernel)

      expect(result).to be_narray_like NArray[[-0.58, 0.37], [-0.53, 1.23]]
    end

    it 'calculates a 3D convolution' do
      # 5x4x3
      a = NArray[
        [[1.0, 0.6, 1.1, 0.2, 0.9], [1.0, 0.7, 0.8, 1.0, 1.0], [0.2, 0.6, 0.1, 0.2, 0.5],
         [0.5, 0.9, 0.2, 0.1, 0.6]],
        [[0.4, 0.9, 0.4, 0.0, 0.6], [0.2, 1.1, 0.2, 0.4, 0.1], [0.4, 0.2, 0.5, 0.8, 0.7],
         [0.1, 0.9, 0.7, 0.1, 0.3]],
        [[0.8, 0.6, 1.0, 0.1, 0.4], [0.3, 0.8, 0.6, 0.7, 1.1], [0.9, 1.0, 0.3, 0.4, 0.6],
         [0.2, 0.5, 0.4, 0.7, 0.2]]
      ]

      # 3x3x3
      b = NArray[
        [[-0.9, 1.2, 0.8], [0.9, 0.1, -0.5], [1.1, 0.1, -1.1]],
        [[-0.2, -1.0, 1.4], [-1.4, 0.0, 1.3], [0.3, 1.0, -0.5]],
        [[0.6, 0.0, 0.7],   [-0.7, 1.1, 1.2], [1.3, 0.7, 0.0]]
      ]

      # Should be 3x2x1
      c = described_class.convolve_basic(a, b)
      expect(c).to be_narray_like NArray[[[5.51, 3.04, 4.3], [3.04, 6.31, 3.87]]]
    end

    it 'calculates a 4D convolution' do
      # 3x4x5x3
      a = NArray[
        [[[0.5, 0.4, 0.9], [0.1, 0.9, 0.8], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]],
         [[0.0, 0.4, 0.0], [0.2, 0.3, 0.8], [0.6, 0.3, 0.2], [0.7, 0.4, 0.3]],
         [[0.3, 0.3, 0.1], [0.6, 0.9, 0.4], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]],
         [[0.0, 0.4, 0.0], [0.2, 0.3, 0.8], [0.6, 0.3, 0.2], [0.7, 0.4, 0.3]],
         [[0.3, 0.3, 0.1], [0.6, 0.9, 0.4], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]]],
        [[[0.5, 0.4, 0.9], [0.1, 0.9, 0.8], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]],
         [[0.0, 0.4, 0.0], [0.2, 0.3, 0.8], [0.6, 0.3, 0.2], [0.7, 0.4, 0.3]],
         [[0.3, 0.3, 0.1], [0.6, 0.9, 0.4], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]],
         [[0.0, 0.4, 0.0], [0.2, 0.3, 0.8], [0.6, 0.3, 0.2], [0.7, 0.4, 0.3]],
         [[0.3, 0.3, 0.1], [0.6, 0.9, 0.4], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]]],
        [[[0.5, 0.4, 0.9], [0.1, 0.9, 0.8], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]],
         [[0.0, 0.4, 0.0], [0.2, 0.3, 0.8], [0.6, 0.3, 0.2], [0.7, 0.4, 0.3]],
         [[0.3, 0.3, 0.1], [0.6, 0.9, 0.4], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]],
         [[0.0, 0.4, 0.0], [0.2, 0.3, 0.8], [0.6, 0.3, 0.2], [0.7, 0.4, 0.3]],
         [[0.3, 0.3, 0.1], [0.6, 0.9, 0.4], [0.4, 0.0, 0.1], [0.8, 0.3, 0.4]]] ]

      # 2x3x3x2
      b = NArray[ [
        [[1.1, 0.6], [1.2, 0.6], [0.8, 0.1]], [[-0.4, 0.8], [0.5, 0.4], [1.2, 0.2]],
        [[0.8, 0.2], [0.5, 0.0], [1.4, 1.3]]
      ],
                  [[[1.1, 0.6], [1.2, 0.6], [0.8, 0.1]], [[-0.4, 0.8], [0.5, 0.4], [1.2, 0.2]],
                   [[0.8, 0.2], [0.5, 0.0], [1.4, 1.3]]] ]

      # Should be 2x2x3x2
      c = described_class.convolve_basic(a, b)
      expect(c).to be_narray_like NArray[
        [[[8.5, 8.2], [11.34, 9.68]], [[7.68, 6.56], [11.24, 7.16]], [[9.14, 6.54], [12.44, 9.2]]],
        [[[8.5, 8.2], [11.34, 9.68]], [[7.68, 6.56], [11.24, 7.16]], [[9.14, 6.54], [12.44, 9.2]]]
      ]
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
