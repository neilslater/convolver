# frozen_string_literal: true

require 'helpers'

# The intentionally direct oracle favors readability over production metrics.
# rubocop:disable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/ParameterLists
module CorrelationReference
  CALCULATION_METHODS = %i[convolve convolve_basic convolve_fft].freeze
  ENTRY_POINTS = (CALCULATION_METHODS + %i[predict_convolve_basic_time predict_convolve_fft_time]).freeze

  module_function

  def calculate(signal, kernel, mode:, boundary: :constant, fill_value: 0.0, origin: 0)
    origins = origin.is_a?(Array) ? origin : Array.new(signal.ndim, origin)
    anchors = kernel.shape.zip(origins).map { |size, shift| (size / 2) + shift }
    result_shape, starts = result_layout(signal.shape, kernel.shape, anchors, mode)
    result = Numo::SFloat.zeros(*result_shape)

    coordinates(result_shape).each do |output_coordinate|
      total = coordinates(kernel.shape).sum do |kernel_coordinate|
        signal_coordinate = output_coordinate.zip(kernel_coordinate, starts).map(&:sum)
        value = boundary_value(signal, signal_coordinate, boundary, fill_value)
        value * kernel[*kernel_coordinate]
      end
      result[*output_coordinate] = total
    end

    result
  end

  def coordinates(shape)
    shape.reduce([[]]) do |coordinates, size|
      coordinates.flat_map { |prefix| size.times.map { |index| prefix + [index] } }
    end
  end

  def result_layout(signal_shape, kernel_shape, anchors, mode)
    case mode
    when :valid
      [signal_shape.zip(kernel_shape).map { |signal, kernel| signal - kernel + 1 }, Array.new(signal_shape.length, 0)]
    when :same
      [signal_shape, anchors.map(&:-@)]
    when :full
      [signal_shape.zip(kernel_shape).map { |signal, kernel| signal + kernel - 1 },
       kernel_shape.map { |kernel| 1 - kernel }]
    end
  end

  def boundary_value(signal, coordinate, boundary, fill_value)
    return signal[*coordinate] if coordinate.zip(signal.shape).all? { |index, size| index.between?(0, size - 1) }
    return fill_value if boundary == :constant

    mapped = coordinate.zip(signal.shape).map { |index, size| boundary_index(index, size, boundary) }
    signal[*mapped]
  end

  def boundary_index(index, size, boundary)
    case boundary
    when :nearest then index.clamp(0, size - 1)
    when :wrap then index % size
    when :reflect
      residue = index % (2 * size)
      residue < size ? residue : (2 * size) - 1 - residue
    when :mirror
      return 0 if size == 1

      period = 2 * (size - 1)
      residue = index % period
      residue < size ? residue : period - residue
    end
  end
end
# rubocop:enable Metrics/AbcSize, Metrics/CyclomaticComplexity, Metrics/MethodLength, Metrics/ParameterLists

describe Convolver do
  CorrelationReference::CALCULATION_METHODS.each do |method_name|
    describe ".#{method_name} output and boundary options" do
      it 'supports valid, same, and full constant output extents' do
        signal = NArray[1, 2, 4]
        kernel = NArray[3, 5]

        valid = described_class.public_send(method_name, signal, kernel)
        same = described_class.public_send(method_name, signal, kernel, mode: :same)
        full = described_class.public_send(method_name, signal, kernel, mode: :full)

        expect(valid).to be_narray_like NArray[13, 26]
        expect(same).to be_narray_like NArray[5, 13, 26]
        expect(full).to be_narray_like NArray[5, 13, 26, 12]
        expect([valid.class, same.class, full.class]).to all(eq(Numo::SFloat))
      end

      it 'uses nonzero constant fill throughout a full result' do
        full = described_class.public_send(
          method_name, NArray[1, 2], NArray[3, 5], mode: :full, fill_value: 10
        )
        same = described_class.public_send(
          method_name, NArray[1, 2, 4], NArray[3, 5], mode: :same, fill_value: -2
        )

        expect(full).to be_narray_like NArray[35, 13, 56]
        expect(same).to be_narray_like NArray[-1, 13, 26]
      end

      it 'implements every same-sized boundary sequence exactly' do
        signal = NArray[1, 2, 4]
        kernel = NArray[1, 10, 100]
        expected = {
          constant: NArray[210, 421, 42],
          nearest: NArray[211, 421, 442],
          reflect: NArray[211, 421, 442],
          mirror: NArray[212, 421, 242],
          wrap: NArray[214, 421, 142]
        }

        expected.each do |boundary, values|
          result = described_class.public_send(method_name, signal, kernel, mode: :same, boundary:)
          expect(result).to be_narray_like values
        end
      end

      it 'applies positive and negative kernel origins' do
        signal = NArray[1, 2, 4, 8]
        kernel = NArray[1, 10, 100]

        positive = described_class.public_send(method_name, signal, kernel, mode: :same, origin: 1)
        negative = described_class.public_send(method_name, signal, kernel, mode: :same, origin: -1)

        expect(positive).to be_narray_like NArray[100, 210, 421, 842]
        expect(negative).to be_narray_like NArray[421, 842, 84, 8]
      end

      it 'supports length-one signals and kernels wider than the signal' do
        signal = NArray[7]
        kernel = NArray[1, 2, 3, 4, 5]

        %i[nearest reflect mirror wrap].each do |boundary|
          result = described_class.public_send(method_name, signal, kernel, mode: :same, boundary:)
          expect(result).to be_narray_like NArray[105]
        end
      end

      it 'maps multidimensional corners and per-axis origins' do
        signal = NArray[[1, 4], [2, 5], [3, 6]].transpose
        kernel = NArray[[1, 3, 5], [2, 4, 6]].transpose
        origin = [1, -1]

        %i[constant nearest reflect mirror wrap].each do |boundary|
          expected = CorrelationReference.calculate(signal, kernel, mode: :same, boundary:, origin:)
          result = described_class.public_send(method_name, signal, kernel, mode: :same, boundary:, origin:)
          expect(result).to be_narray_like expected
        end
      end

      it 'handles periodic folding on odd shapes with a larger kernel' do
        signal = NArray[2, 3, 5, 7, 11]
        kernel = NArray[11, 13, 17, 19, 23, 29, 31]
        expected = CorrelationReference.calculate(
          signal, kernel, mode: :same, boundary: :wrap, origin: 2
        )

        result = described_class.public_send(
          method_name, signal, kernel, mode: :same, boundary: :wrap, origin: 2
        )

        expect(result).to be_narray_like expected
      end
    end
  end

  describe 'shared option validation' do
    let(:validation_signal) { NArray[1, 2, 4] }
    let(:validation_kernel) { NArray[1, 10, 100] }

    invalid_options = [
      [{ mode: 'same' }, /mode must be one of/],
      [{ mode: :unknown }, /mode must be one of/],
      [{ mode: :same, boundary: 'wrap' }, /boundary must be one of/],
      [{ mode: :same, boundary: :unknown }, /boundary must be one of/],
      [{ boundary: :reflect }, /mode: :valid only supports boundary: :constant/],
      [{ fill_value: 1 }, /mode: :valid requires fill_value: 0/],
      [{ mode: :same, boundary: :reflect, fill_value: 0 }, /fill_value is only supported/],
      [{ mode: :full, boundary: :wrap }, /mode: :full only supports boundary: :constant/],
      [{ origin: 1 }, /nonzero origin is only supported with mode: :same/],
      [{ mode: :full, origin: -1 }, /nonzero origin is only supported with mode: :same/],
      [{ mode: :same, origin: 'left' }, /origin must be an Integer/],
      [{ mode: :same, origin: [0, 0] }, /origin must contain one Integer per dimension/],
      [{ mode: :same, origin: ['left'] }, /origin must contain one Integer per dimension/],
      [{ mode: :same, origin: -2 }, /origin -2 is out of range/],
      [{ mode: :same, origin: 2 }, /origin 2 is out of range/],
      [{ mode: :same, fill_value: Complex(1, 2) }, /fill_value must be a real numeric value/]
    ]

    CorrelationReference::ENTRY_POINTS.each do |entry_point|
      invalid_options.each do |options, error|
        it "rejects #{options.inspect} through .#{entry_point}" do
          expect { described_class.public_send(entry_point, validation_signal, validation_kernel, **options) }
            .to raise_error(ArgumentError, error)
        end
      end
    end

    it 'allows large kernels for same and full through every entry point' do
      small_signal = NArray[1, 2]
      large_kernel = NArray[1, 2, 3]
      all_entry_points = %i[
        convolve convolve_basic convolve_fft predict_convolve_basic_time predict_convolve_fft_time
      ]

      all_entry_points.each do |entry_point|
        expect { described_class.public_send(entry_point, small_signal, large_kernel, mode: :same) }
          .not_to raise_error
        expect { described_class.public_send(entry_point, small_signal, large_kernel, mode: :full) }
          .not_to raise_error
      end
    end

    it 'accepts explicitly supplied defaults for valid mode through every entry point' do
      CorrelationReference::ENTRY_POINTS.each do |entry_point|
        expect do
          described_class.public_send(
            entry_point, validation_signal, validation_kernel,
            mode: :valid, boundary: :constant, fill_value: 0, origin: [0]
          )
        end.not_to raise_error
      end
    end
  end

  describe 'cross-implementation coverage' do
    it 'matches an independent reference across randomized modes, boundaries, and ranks' do
      random = Random.new(12_345)
      cases = [
        [[4], [3], :valid, :constant, 0.0, 0],
        [[4], [3], :full, :constant, -0.75, 0],
        [[4], [5], :same, :constant, 1.25, -1],
        *%i[nearest reflect mirror wrap].map { |boundary| [[4], [5], :same, boundary, 0.0, -1] },
        [[2, 3], [3, 2], :full, :constant, -0.75, 0],
        [[2, 3], [3, 2], :same, :constant, 1.25, [1, -1]],
        *%i[nearest reflect mirror wrap].map do |boundary|
          [[2, 3], [3, 2], :same, boundary, 0.0, [1, -1]]
        end
      ]

      cases.each do |case_data|
        signal_shape, kernel_shape, mode, boundary, fill_value, origin = case_data
        signal = NArray[*Array.new(signal_shape.inject(:*)) { random.rand(-2.0..3.0) }].reshape(*signal_shape)
        kernel = NArray[*Array.new(kernel_shape.inject(:*)) { random.rand(-3.0..2.0) }].reshape(*kernel_shape)
        expected = CorrelationReference.calculate(signal, kernel, mode:, boundary:, fill_value:, origin:)
        options = { mode:, boundary:, origin: }
        options[:fill_value] = fill_value if boundary == :constant

        expect(described_class.convolve_basic(signal, kernel, **options)).to be_narray_like(expected, 1e-7)
        expect(described_class.convolve_fft(signal, kernel, **options)).to be_narray_like(expected, 1e-7)
      end
    end

    it 'supports boundary extension on higher-rank inputs' do
      signal = NArray[*Array(1..8)].reshape(2, 2, 2)
      kernel = NArray[*Array(1..12)].reshape(3, 2, 2)
      expected = CorrelationReference.calculate(
        signal, kernel, mode: :same, boundary: :mirror, origin: [1, -1, 0]
      )

      CorrelationReference::CALCULATION_METHODS.each do |method_name|
        result = described_class.public_send(
          method_name, signal, kernel, mode: :same, boundary: :mirror, origin: [1, -1, 0]
        )
        expect(result).to be_narray_like expected
      end
    end

    it 'exposes the keyword API from each calculation and estimator method' do
      public_methods = CorrelationReference::CALCULATION_METHODS + %i[
        convolve_fftw3 predict_convolve_basic_time predict_convolve_fft_time
      ]
      expected_parameters = [
        %i[req signal], %i[req kernel], %i[key mode], %i[key boundary],
        %i[key fill_value], %i[key origin]
      ]

      public_methods.each do |method_name|
        expect(described_class.method(method_name).parameters).to eq expected_parameters
      end
    end
  end

  describe 'algorithm selection and estimation' do
    it 'forwards the complete requested semantics to the selected implementation' do
      signal = NArray[1, 2, 4]
      kernel = NArray[1, 10, 100]
      allow(described_class).to receive(:convolve_basic).and_call_original

      described_class.convolve(signal, kernel, mode: :same, boundary: :mirror, origin: -1)

      expect(described_class).to have_received(:convolve_basic)
        .with(signal, kernel, mode: :same, boundary: :mirror, origin: -1)
    end

    it 'accounts for extension and periodic transform sizes' do
      signal = NArray.ones(9)
      kernel = NArray.ones(5)

      valid_basic = described_class.predict_convolve_basic_time(signal, kernel)
      same_basic = described_class.predict_convolve_basic_time(signal, kernel, mode: :same)
      same_fft = described_class.predict_convolve_fft_time(signal, kernel, mode: :same)
      wrap_fft = described_class.predict_convolve_fft_time(signal, kernel, mode: :same, boundary: :wrap)

      expect(same_basic).to be > valid_basic
      expect(wrap_fft).to be > 0
      expect(wrap_fft).to be < same_fft
    end
  end

  it 'keeps the native valid primitive private' do
    expect(described_class.private_methods).to include(:convolve_basic_valid)
    expect(described_class).not_to respond_to(:convolve_basic_valid)
  end
end
