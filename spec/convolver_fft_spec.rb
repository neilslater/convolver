# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe '.convolve_fft' do
    it 'works like the example in the README' do
      a = NArray[0.3, 0.4, 0.5]
      b = NArray[1.3, -0.5]
      c = described_class.convolve_fft(a, b)
      expect(c).to be_narray_like NArray[0.19, 0.27]
    end

    one_dimensional_cases = [
      [[0.3], [-0.7], [-0.21]],
      [[0.3, 0.4, 0.5, 0.2], [-0.7], [-0.21, -0.28, -0.35, -0.14]],
      [[0.3, 0.4, 0.5, 0.2], [1.1, -0.7], [0.05, 0.09, 0.41]],
      [[0.3, 0.4, 0.5, 0.2], [1.1, -0.7, -0.2], [-0.05, 0.05]],
      [[0.3, 0.4, 0.5, 0.2, 0.6], [1.1, -0.7], [0.05, 0.09, 0.41, -0.2]],
      [[0.3, 0.4, 0.5, 0.2, 0.6], [1.1, -0.7, 2.1], [1.1, 0.51, 1.67]],
      [[0.3, 0.4, 0.5, 0.2, 0.6], [0.6, -0.5, -0.4, 0.7], [-0.08, 0.33]]
    ]

    one_dimensional_cases.each_with_index do |(signal, kernel, expected), index|
      it "convolves 1D signal and kernel case #{index + 1}" do
        expect(described_class.convolve_fft(NArray[*signal], NArray[*kernel])).to be_narray_like NArray[*expected]
      end
    end

    it 'calculates a 2D convolution' do
      a = NArray[[0.3, 0.4, 0.5], [0.6, 0.8, 0.2], [0.9, 1.0, 0.1]]
      b = NArray[[1.2, -0.5], [0.5, -1.3]]
      c = described_class.convolve_fft(a, b)
      expect(c).to be_narray_like NArray[[-0.58, 0.37], [-0.53, 1.23]]
    end

    it 'calculates a 3D convolution' do
      result = described_class.convolve_fft(ConvolutionFixtures.three_dimensional_signal,
                                            ConvolutionFixtures.three_dimensional_kernel)
      expect(result).to be_narray_like ConvolutionFixtures.three_dimensional_result
    end

    it 'calculates a 4D convolution' do
      result = described_class.convolve_fft(ConvolutionFixtures.four_dimensional_signal,
                                            ConvolutionFixtures.four_dimensional_kernel)
      expect(result).to be_narray_like ConvolutionFixtures.four_dimensional_result
    end

    describe 'compared with .convolve_basic' do
      it 'produces same results for 1D arrays' do
        expect_one_dimensional_fft_to_match_basic
      end

      it 'produces same results for 2D arrays' do
        expect_two_dimensional_fft_to_match_basic
      end

      it 'produces same results for 3D arrays' do
        expect_three_dimensional_fft_to_match_basic
      end
    end

    def expect_one_dimensional_fft_to_match_basic
      (1..30).each do |signal_length|
        (1..signal_length).each { |kernel_length| expect_fft_to_match_basic([signal_length], [kernel_length]) }
      end
    end

    def expect_two_dimensional_fft_to_match_basic
      (3..10).each do |signal_x|
        ((signal_x - 2)..(signal_x + 2)).each do |signal_y|
          (1..signal_x).to_a.product((1..signal_y).to_a).each do |kernel_shape|
            expect_fft_to_match_basic([signal_x, signal_y], kernel_shape)
          end
        end
      end
    end

    def expect_three_dimensional_fft_to_match_basic
      (3..5).each do |signal_x|
        neighbors = ((signal_x - 2)..(signal_x + 2)).to_a
        neighbors.product(neighbors).each do |signal_y, signal_z|
          kernel_ranges = [signal_x, signal_y, signal_z].map { |size| (1..size).to_a }
          kernel_ranges.first.product(*kernel_ranges.drop(1)).each do |kernel_shape|
            expect_fft_to_match_basic([signal_x, signal_y, signal_z], kernel_shape)
          end
        end
      end
    end

    def expect_fft_to_match_basic(signal_shape, kernel_shape)
      signal = NArray.sfloat(*signal_shape).random
      kernel = NArray.sfloat(*kernel_shape).random
      expected = described_class.convolve_basic(signal, kernel)
      expect(described_class.convolve_fft(signal, kernel)).to be_narray_like expected
    end
  end
end
