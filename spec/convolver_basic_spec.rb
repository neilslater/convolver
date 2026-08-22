# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe '.correlate_basic' do
    it 'preserves the version 2 valid cross-correlation behavior' do
      result = described_class.correlate_basic(
        ConvolutionFixtures.one_dimensional_signal, ConvolutionFixtures.one_dimensional_kernel
      )
      expect(result).to be_narray_like NArray[13, 26]
    end

    it 'calculates an asymmetric 2D cross-correlation' do
      result = described_class.correlate_basic(
        ConvolutionFixtures.two_dimensional_signal, ConvolutionFixtures.two_dimensional_kernel
      )
      expect(result).to be_narray_like NArray[[-0.58, 0.37], [-0.53, 1.23]]
    end

    it 'calculates rectangular, 3D, and 4D fixture correlations' do
      results = [
        described_class.correlate_basic(CorrelationFixtures.rectangular_signal,
                                        CorrelationFixtures.rectangular_kernel),
        described_class.correlate_basic(CorrelationFixtures.three_dimensional_signal,
                                        CorrelationFixtures.three_dimensional_kernel),
        described_class.correlate_basic(CorrelationFixtures.four_dimensional_signal,
                                        CorrelationFixtures.four_dimensional_kernel)
      ]
      expected = [CorrelationFixtures.rectangular_result, CorrelationFixtures.three_dimensional_result,
                  CorrelationFixtures.four_dimensional_result]
      expect(results.zip(expected)).to all(satisfy { |result, fixture| be_narray_like(fixture).matches?(result) })
    end
  end

  describe '.convolve_basic' do
    it 'calculates mathematical convolution without a reversed-kernel buffer' do
      result = described_class.convolve_basic(
        ConvolutionFixtures.one_dimensional_signal, ConvolutionFixtures.one_dimensional_kernel
      )
      expect(result).to be_narray_like ConvolutionFixtures.one_dimensional_valid
    end

    it 'calculates an asymmetric 2D mathematical convolution' do
      result = described_class.convolve_basic(
        ConvolutionFixtures.two_dimensional_signal, ConvolutionFixtures.two_dimensional_kernel
      )
      expect(result).to be_narray_like ConvolutionFixtures.two_dimensional_valid
    end
  end

  %i[correlate_basic convolve_basic].each do |method_name|
    describe ".#{method_name} native input handling" do
      it 'calculates from non-contiguous views' do
        signal = NArray[[0.3, 0.6, 0.9], [0.4, 0.8, 1.0], [0.5, 0.2, 0.1]].transpose
        kernel = NArray[[1.2, 0.5], [-0.5, -1.3]].transpose
        operation = method_name == :correlate_basic ? :correlation : :convolution
        expected = OperationReference.calculate(operation, signal, kernel, mode: :valid)

        expect(described_class.public_send(method_name, signal, kernel)).to be_narray_like expected
      end

      it 'rejects arrays with different ranks' do
        expect { described_class.public_send(method_name, NArray.sfloat(2), NArray.sfloat(2, 2)) }
          .to raise_error(ArgumentError, 'signal and kernel must have equal rank')
      end

      it 'rejects values that are not Numo arrays' do
        expect { described_class.public_send(method_name, [1.0], NArray[1.0]) }
          .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
      end

      it 'rejects empty arrays' do
        expect { described_class.public_send(method_name, NArray.zeros(0), NArray[1.0]) }
          .to raise_error(ArgumentError, 'signal and kernel must not be empty')
      end

      it 'rejects arrays above the maximum supported rank' do
        rank_17_array = NArray.sfloat(*([1] * 17))
        expect { described_class.public_send(method_name, rank_17_array, rank_17_array) }
          .to raise_error(ArgumentError, 'maximum supported rank is 16')
      end

      it 'calculates at the maximum supported rank' do
        rank_16_shape = Array.new(16, 1)
        signal = NArray.ones(*rank_16_shape) * 2
        kernel = NArray.ones(*rank_16_shape) * 3

        expect(described_class.public_send(method_name, signal, kernel))
          .to be_narray_like NArray.ones(*rank_16_shape) * 6
      end

      it 'rejects a kernel larger than the signal in valid mode' do
        expect { described_class.public_send(method_name, NArray.sfloat(2), NArray.sfloat(3)) }
          .to raise_error(ArgumentError, 'kernel must not be larger than signal in any dimension')
      end
    end
  end
end
