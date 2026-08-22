# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe '.correlate_fft' do
    it 'preserves the version 2 valid cross-correlation behavior' do
      result = described_class.correlate_fft(
        ConvolutionFixtures.one_dimensional_signal, ConvolutionFixtures.one_dimensional_kernel
      )
      expect(result).to be_narray_like NArray[13, 26]
    end

    it 'preserves the higher-dimensional correlation fixtures' do
      results = [
        described_class.correlate_fft(CorrelationFixtures.three_dimensional_signal,
                                      CorrelationFixtures.three_dimensional_kernel),
        described_class.correlate_fft(CorrelationFixtures.four_dimensional_signal,
                                      CorrelationFixtures.four_dimensional_kernel)
      ]
      expected = [CorrelationFixtures.three_dimensional_result, CorrelationFixtures.four_dimensional_result]
      expect(results.zip(expected)).to all(satisfy { |result, fixture| be_narray_like(fixture).matches?(result) })
    end
  end

  describe '.convolve_fft' do
    it 'calculates the literal asymmetric convolution fixture' do
      result = described_class.convolve_fft(
        ConvolutionFixtures.one_dimensional_signal, ConvolutionFixtures.one_dimensional_kernel
      )
      expect(result).to be_narray_like ConvolutionFixtures.one_dimensional_valid
    end

    it 'calculates asymmetric 2D mathematical convolution' do
      result = described_class.convolve_fft(
        ConvolutionFixtures.two_dimensional_signal, ConvolutionFixtures.two_dimensional_kernel
      )
      expect(result).to be_narray_like ConvolutionFixtures.two_dimensional_valid
    end
  end

  OperationReference::OPERATIONS.each do |operation|
    fft_method = OperationReference::CALCULATION_METHODS.fetch(operation).last
    basic_method = OperationReference::CALCULATION_METHODS.fetch(operation)[1]

    describe ".#{fft_method} compared with .#{basic_method}" do
      it 'matches for 1D arrays, including odd full-transform lengths' do
        (1..30).each do |signal_length|
          (1..signal_length).each do |kernel_length|
            expect_fft_to_match_basic(operation, [signal_length], [kernel_length])
          end
        end
      end

      it 'matches for rectangular 2D arrays' do
        expect_rectangular_fft_to_match_basic(operation)
      end

      it 'matches selected 3D arrays' do
        expect_random_fft_to_match_basic(operation)
      end
    end
  end

  describe 'real transform shape selection' do
    it 'uses the smallest valid even final axis when its factors are competitive' do
      plan, kernel_shape = fft_plan([60, 60, 60], [4, 4, 4])
      expect(plan.linear_fft_shape(kernel_shape)).to eq [63, 63, 64]
    end

    it 'uses a fast real shape instead of retaining large prime factors' do
      plan, kernel_shape = fft_plan([192, 256], [15, 18])
      expect(plan.linear_fft_shape(kernel_shape)).to eq [216, 288]
    end

    it 'rejects linear FFT shape overflow before allocation' do
      shapes_class = described_class.const_get(:OperationShapes, false)
      shapes = shapes_class.new([shapes_class::SIZE_MAX], [2],
                                operation: :correlation, mode: :valid, anchors: [1])
      expect { shapes.linear_fft_shape([2]) }
        .to raise_error(RangeError, 'FFT shape exceeds native implementation limit')
    end
  end

  def expect_fft_to_match_basic(operation, signal_shape, kernel_shape)
    signal = NArray.sfloat(*signal_shape).random
    kernel = NArray.sfloat(*kernel_shape).random
    methods = OperationReference::CALCULATION_METHODS.fetch(operation)
    expected = described_class.public_send(methods[1], signal, kernel)
    expect(described_class.public_send(methods[2], signal, kernel)).to be_narray_like expected
  end

  def expect_rectangular_fft_to_match_basic(operation)
    (3..9).each do |signal_x|
      ((signal_x - 2)..(signal_x + 2)).each do |signal_y|
        kernel_shapes(signal_x, signal_y).each do |kernel_shape|
          expect_fft_to_match_basic(operation, [signal_x, signal_y], kernel_shape)
        end
      end
    end
  end

  def kernel_shapes(signal_x, signal_y)
    (1..signal_x).to_a.product((1..signal_y).to_a)
  end

  def expect_random_fft_to_match_basic(operation)
    random = Random.new(operation == :correlation ? 12_345 : 54_321)
    100.times do
      signal_shape = Array.new(3) { random.rand(2..8) }
      kernel_shape = signal_shape.map { |size| random.rand(1..size) }
      expect_fft_to_match_basic(operation, signal_shape, kernel_shape)
    end
  end

  def fft_plan(signal_shape, kernel_shape)
    signal = NArray.zeros(*signal_shape)
    kernel = NArray.zeros(*kernel_shape)
    plan_class = described_class.const_get(:OperationPlan, false)
    plan = plan_class.new(
      signal, kernel, operation: :correlation, mode: :valid,
                      boundary: :constant, fill_value: 0.0, origin: 0
    )
    [plan, kernel_shape]
  end
end
