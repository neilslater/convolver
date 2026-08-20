# frozen_string_literal: true

require 'helpers'

module CorrelationReference
  CALCULATION_METHODS = %i[convolve convolve_basic convolve_fft].freeze
  ENTRY_POINTS = (CALCULATION_METHODS + %i[predict_convolve_basic_time predict_convolve_fft_time]).freeze

  # Evaluates one correlation independently of Convolver's planning code.
  class Calculation
    def initialize(signal, kernel, mode:, boundary:, fill_value:, origin:)
      @signal = signal
      @kernel = kernel
      @boundary = boundary
      @fill_value = fill_value
      origins = origin.is_a?(Array) ? origin : Array.new(signal.ndim, origin)
      anchors = kernel.shape.zip(origins).map { |size, shift| (size / 2) + shift }
      @result_shape, @starts = CorrelationReference.result_layout(signal.shape, kernel.shape, anchors, mode)
    end

    def call
      result = Numo::SFloat.zeros(*result_shape)
      CorrelationReference.coordinates(result_shape).each do |output_coordinate|
        result[*output_coordinate] = correlation_at(output_coordinate)
      end
      result
    end

    private

    attr_reader :signal, :kernel, :boundary, :fill_value, :result_shape, :starts

    def correlation_at(output_coordinate)
      CorrelationReference.coordinates(kernel.shape).sum do |kernel_coordinate|
        signal_coordinate = output_coordinate.zip(kernel_coordinate, starts).map(&:sum)
        value = CorrelationReference.boundary_value(signal, signal_coordinate, boundary, fill_value)
        value * kernel[*kernel_coordinate]
      end
    end
  end

  module_function

  def calculate(signal, kernel, mode:, boundary: :constant, fill_value: 0.0, origin: 0)
    Calculation.new(signal, kernel, mode:, boundary:, fill_value:, origin:).call
  end

  def coordinates(shape)
    shape.reduce([[]]) do |coordinates, size|
      coordinates.flat_map { |prefix| size.times.map { |index| prefix + [index] } }
    end
  end

  def result_layout(signal_shape, kernel_shape, anchors, mode)
    case mode
    when :valid then valid_layout(signal_shape, kernel_shape)
    when :same then [signal_shape, anchors.map(&:-@)]
    when :full then full_layout(signal_shape, kernel_shape)
    end
  end

  def valid_layout(signal_shape, kernel_shape)
    result_shape = signal_shape.zip(kernel_shape).map { |signal, kernel| signal - kernel + 1 }
    [result_shape, Array.new(signal_shape.length, 0)]
  end

  def full_layout(signal_shape, kernel_shape)
    result_shape = signal_shape.zip(kernel_shape).map { |signal, kernel| signal + kernel - 1 }
    [result_shape, kernel_shape.map { |kernel| 1 - kernel }]
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
    when :reflect then reflect_index(index, size)
    when :mirror then mirror_index(index, size)
    end
  end

  def reflect_index(index, size)
    residue = index % (2 * size)
    residue < size ? residue : (2 * size) - 1 - residue
  end

  def mirror_index(index, size)
    return 0 if size == 1

    period = 2 * (size - 1)
    residue = index % period
    residue < size ? residue : period - residue
  end
end

describe Convolver do
  CorrelationReference::CALCULATION_METHODS.each do |method_name|
    describe ".#{method_name} output and boundary options" do
      extent_cases = {
        valid: [NArray[13, 26], {}],
        same: [NArray[5, 13, 26], { mode: :same }],
        full: [NArray[5, 13, 26, 12], { mode: :full }]
      }
      extent_cases.each do |mode, (expected, options)|
        it "supports the #{mode} constant output extent as SFloat" do
          result = described_class.public_send(method_name, NArray[1, 2, 4], NArray[3, 5], **options)
          expect(result).to be_a(Numo::SFloat).and be_narray_like expected
        end
      end

      constant_fill_cases = [
        [NArray[1, 2], NArray[3, 5], { mode: :full, fill_value: 10 }, NArray[35, 13, 56]],
        [NArray[1, 2, 4], NArray[3, 5], { mode: :same, fill_value: -2 }, NArray[-1, 13, 26]]
      ]
      constant_fill_cases.each do |signal, kernel, options, expected|
        it "uses nonzero constant fill throughout a #{options.fetch(:mode)} result" do
          result = described_class.public_send(method_name, signal, kernel, **options)
          expect(result).to be_narray_like expected
        end
      end

      boundary_cases = {
        constant: NArray[210, 421, 42],
        nearest: NArray[211, 421, 442],
        reflect: NArray[211, 421, 442],
        mirror: NArray[212, 421, 242],
        wrap: NArray[214, 421, 142]
      }
      boundary_cases.each do |boundary, expected|
        it "implements the #{boundary} same-sized boundary sequence" do
          signal = NArray[1, 2, 4]
          kernel = NArray[1, 10, 100]
          result = described_class.public_send(method_name, signal, kernel, mode: :same, boundary:)
          expect(result).to be_narray_like expected
        end
      end

      { 1 => NArray[100, 210, 421, 842], -1 => NArray[421, 842, 84, 8] }.each do |origin, expected|
        it "applies a kernel origin of #{origin}" do
          result = described_class.public_send(
            method_name, NArray[1, 2, 4, 8], NArray[1, 10, 100], mode: :same, origin:
          )
          expect(result).to be_narray_like expected
        end
      end

      %i[nearest reflect mirror wrap].each do |boundary|
        it "supports a length-one signal and wider kernel with #{boundary}" do
          result = described_class.public_send(
            method_name, NArray[7], NArray[1, 2, 3, 4, 5], mode: :same, boundary:
          )
          expect(result).to be_narray_like NArray[105]
        end
      end

      %i[constant nearest reflect mirror wrap].each do |boundary|
        it "maps multidimensional #{boundary} corners and per-axis origins" do
          signal = NArray[[1, 4], [2, 5], [3, 6]].transpose
          kernel = NArray[[1, 3, 5], [2, 4, 6]].transpose
          expected = CorrelationReference.calculate(signal, kernel, mode: :same, boundary:, origin: [1, -1])
          result = described_class.public_send(method_name, signal, kernel, mode: :same, boundary:, origin: [1, -1])
          expect(result).to be_narray_like expected
        end
      end

      it 'handles periodic folding on odd shapes with a larger kernel' do
        expect_periodic_folding_to_match_reference(method_name)
      end
    end
  end

  def expect_periodic_folding_to_match_reference(method_name)
    signal = NArray[2, 3, 5, 7, 11]
    kernel = NArray[11, 13, 17, 19, 23, 29, 31]
    options = { mode: :same, boundary: :wrap, origin: 2 }
    expected = CorrelationReference.calculate(signal, kernel, **options)
    expect(described_class.public_send(method_name, signal, kernel, **options)).to be_narray_like expected
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

    %i[same full].each do |mode|
      it "allows large kernels for #{mode} through every entry point" do
        CorrelationReference::ENTRY_POINTS.each do |entry_point|
          expect { described_class.public_send(entry_point, NArray[1, 2], NArray[1, 2, 3], mode:) }
            .not_to raise_error
        end
      end
    end

    it 'accepts explicitly supplied defaults for valid mode through every entry point' do
      expect_explicit_defaults_to_be_valid
    end

    def expect_explicit_defaults_to_be_valid
      CorrelationReference::ENTRY_POINTS.each do |entry_point|
        options = { mode: :valid, boundary: :constant, fill_value: 0, origin: [0] }
        expect { described_class.public_send(entry_point, validation_signal, validation_kernel, **options) }
          .not_to raise_error
      end
    end
  end

  describe 'cross-implementation coverage' do
    it 'matches an independent reference across randomized modes, boundaries, and ranks' do
      expect_randomized_cases_to_match_reference
    end

    it 'supports boundary extension on higher-rank inputs' do
      expect_higher_rank_extensions_to_match_reference
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

    def expect_randomized_cases_to_match_reference
      random = Random.new(12_345)
      randomized_cases.each { |case_data| expect_randomized_case_to_match_reference(case_data, random) }
    end

    def randomized_cases
      [
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
    end

    def expect_randomized_case_to_match_reference(case_data, random)
      signal, kernel = random_arrays(case_data.first(2), random)
      options = case_options(case_data.drop(2))
      expected = CorrelationReference.calculate(signal, kernel, **options, fill_value: case_data[4])
      results = %i[convolve_basic convolve_fft].map do |method_name|
        described_class.public_send(method_name, signal, kernel, **options)
      end
      expect(results).to all(be_narray_like(expected, 1e-7))
    end

    def random_arrays(shapes, random)
      ranges = [-2.0..3.0, -3.0..2.0]
      shapes.zip(ranges).map do |shape, range|
        NArray[*Array.new(shape.inject(:*)) { random.rand(range) }].reshape(*shape)
      end
    end

    def case_options(case_data)
      mode, boundary, fill_value, origin = case_data
      { mode:, boundary:, origin: }.tap do |options|
        options[:fill_value] = fill_value if boundary == :constant
      end
    end

    def expect_higher_rank_extensions_to_match_reference
      signal = NArray[*Array(1..8)].reshape(2, 2, 2)
      kernel = NArray[*Array(1..12)].reshape(3, 2, 2)
      options = { mode: :same, boundary: :mirror, origin: [1, -1, 0] }
      expected = CorrelationReference.calculate(signal, kernel, **options)
      results = CorrelationReference::CALCULATION_METHODS.map do |method_name|
        described_class.public_send(method_name, signal, kernel, **options)
      end
      expect(results).to all(be_narray_like(expected))
    end
  end

  describe 'algorithm selection and estimation' do
    let(:selection_signal) { NArray[1, 2, 4] }
    let(:selection_kernel) { NArray[1, 10, 100] }
    let(:estimates) do
      signal = NArray.ones(9)
      kernel = NArray.ones(5)
      {
        valid_basic: described_class.predict_convolve_basic_time(signal, kernel),
        same_basic: described_class.predict_convolve_basic_time(signal, kernel, mode: :same),
        same_fft: described_class.predict_convolve_fft_time(signal, kernel, mode: :same),
        wrap_fft: described_class.predict_convolve_fft_time(signal, kernel, mode: :same, boundary: :wrap)
      }
    end

    before do
      allow(described_class).to receive(:convolve_basic).and_call_original
      described_class.convolve(selection_signal, selection_kernel, mode: :same, boundary: :mirror, origin: -1)
    end

    it 'forwards the complete requested semantics to the selected implementation' do
      expect(described_class).to have_received(:convolve_basic)
        .with(selection_signal, selection_kernel, mode: :same, boundary: :mirror, origin: -1)
    end

    it 'accounts for extension in direct estimates' do
      expect(estimates.fetch(:same_basic)).to be > estimates.fetch(:valid_basic)
    end

    it 'produces a positive periodic transform estimate' do
      expect(estimates.fetch(:wrap_fft)).to be_positive
    end

    it 'estimates the periodic transform below the linear transform' do
      expect(estimates.fetch(:wrap_fft)).to be < estimates.fetch(:same_fft)
    end
  end

  describe 'the native valid primitive' do
    it 'is private' do
      expect(described_class.private_methods).to include(:convolve_basic_valid)
    end

    it 'is not publicly callable' do
      expect(described_class).not_to respond_to(:convolve_basic_valid)
    end
  end
end
