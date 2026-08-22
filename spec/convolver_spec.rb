# frozen_string_literal: true

require 'helpers'

describe Convolver do
  {
    correlation: { method: :correlate, basic: :correlate_basic, fft: :correlate_fft,
                   expected: Numo::SFloat[0.19, 0.27] },
    convolution: { method: :convolve, basic: :convolve_basic, fft: :convolve_fft,
                   expected: Numo::SFloat[0.37, 0.45] }
  }.each_value do |methods|
    describe ".#{methods.fetch(:method)}" do
      it 'works like its README example' do
        signal = Numo::SFloat[0.3, 0.4, 0.5]
        kernel = Numo::SFloat[1.3, -0.5]
        result = described_class.public_send(methods.fetch(:method), signal, kernel)
        expect(result).to be_narray_like methods.fetch(:expected)
      end

      it 'matches the direct implementation across different sizes' do
        expect_automatic_operations_to_match_basic(methods)
      end

      context 'with small inputs' do
        before { exercise_algorithm_selection(methods, 50, 10) }

        it 'chooses the direct implementation' do
          expect(described_class).to have_received(methods.fetch(:basic)).once
        end

        it 'does not choose the FFT implementation' do
          expect(described_class).not_to have_received(methods.fetch(:fft))
        end
      end

      context 'with large inputs' do
        before { exercise_algorithm_selection(methods, 500, 100) }

        it 'chooses the FFT implementation' do
          expect(described_class).to have_received(methods.fetch(:fft)).once
        end

        it 'does not choose the direct implementation' do
          expect(described_class).not_to have_received(methods.fetch(:basic))
        end
      end

      it 'rejects values that are not Numo arrays' do
        expect { described_class.public_send(methods.fetch(:method), [1.0], Numo::SFloat[1.0]) }
          .to raise_error(ArgumentError, 'signal and kernel must be Numo::NArray values')
      end

      it 'rejects arrays with different ranks' do
        expect do
          described_class.public_send(methods.fetch(:method), Numo::SFloat.zeros(2), Numo::SFloat.zeros(2, 2))
        end.to raise_error(ArgumentError, 'signal and kernel must have equal rank')
      end

      it 'rejects a kernel larger than the signal in valid mode' do
        expect do
          described_class.public_send(methods.fetch(:method), Numo::SFloat.zeros(2), Numo::SFloat.zeros(3))
        end.to raise_error(ArgumentError, 'kernel must not be larger than signal in any dimension')
      end
    end
  end

  it 'removes the deprecated FFTW3 compatibility method' do
    expect(described_class).not_to respond_to(:convolve_fftw3)
  end

  it 'raises NoMethodError for the removed FFTW3 compatibility method' do
    expect { described_class.convolve_fftw3(Numo::SFloat[1], Numo::SFloat[1]) }
      .to raise_error(NoMethodError)
  end

  it 'does not introduce cross_correlate aliases' do
    aliases = %i[cross_correlate cross_correlate_basic cross_correlate_fft
                 predict_cross_correlate_basic_time predict_cross_correlate_fft_time]
    expect(described_class).not_to respond_to(*aliases)
  end

  def expect_automatic_operations_to_match_basic(methods)
    signal_sizes = [10, 30, 60, 90, 100, 120, 130, 150, 175, 200]
    kernel_sizes = [5, 10, 12, 15, 20, 30, 40, 50]
    valid_size_pairs(signal_sizes, kernel_sizes).each { |sizes| expect_automatic_case_to_match(methods, sizes) }
  end

  def valid_size_pairs(signal_sizes, kernel_sizes)
    signal_sizes.product(kernel_sizes).select { |signal_size, kernel_size| kernel_size < signal_size }
  end

  def expect_automatic_case_to_match(methods, sizes)
    signal, kernel = sizes.map { |size| random_square(size) }
    expected = invoke_operation(methods, :basic, signal, kernel)
    result = invoke_operation(methods, :method, signal, kernel)
    expect(result).to be_narray_like(expected, 1e-6)
  end

  def invoke_operation(methods, implementation, signal, kernel)
    described_class.public_send(methods.fetch(implementation), signal, kernel)
  end

  def exercise_algorithm_selection(methods, signal_size, kernel_size)
    allow(described_class).to receive(methods.fetch(:basic))
    allow(described_class).to receive(methods.fetch(:fft))
    signal = random_square(signal_size)
    kernel = random_square(kernel_size)
    described_class.public_send(methods.fetch(:method), signal, kernel)
  end

  def random_square(size)
    Numo::SFloat.new(size, size).rand
  end
end
