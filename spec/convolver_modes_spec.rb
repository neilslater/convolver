# frozen_string_literal: true

require 'helpers'

describe Convolver do
  OperationReference::OPERATIONS.each do |operation|
    OperationReference::CALCULATION_METHODS.fetch(operation).each do |method_name|
      describe ".#{method_name} output and boundary options" do
        it 'returns the literal valid, same, and full asymmetric fixtures as SFloat' do
          expect_extent_fixtures_to_match(operation, method_name)
        end

        it 'uses nonzero constant fill throughout full and same results' do
          cases = [
            [NArray[1, 2], NArray[3, 5], { mode: :full, fill_value: 10.0 }],
            [NArray[1, 2, 4], NArray[3, 5], { mode: :same, fill_value: -2.0 }]
          ]
          expect_cases_to_match_reference(operation, method_name, cases)
        end

        %i[constant nearest reflect mirror wrap].each do |boundary|
          it "implements the #{boundary} same-sized boundary sequence" do
            signal = NArray[1, 2, 4]
            kernel = NArray[1, 10, 100]
            options = { mode: :same, boundary: }
            expect_to_match_reference(operation, method_name, signal, kernel, options)
          end
        end

        [-1, 1].each do |origin|
          it "applies a kernel origin of #{origin}" do
            signal = NArray[1, 2, 4, 8]
            kernel = NArray[1, 10, 100]
            expect_to_match_reference(operation, method_name, signal, kernel, mode: :same, origin:)
          end
        end

        %i[nearest reflect mirror wrap].each do |boundary|
          it "supports a length-one signal and wider kernel with #{boundary}" do
            signal = NArray[7]
            kernel = NArray[1, 2, 3, 4, 5]
            expect_to_match_reference(operation, method_name, signal, kernel, mode: :same, boundary:)
          end
        end

        it 'maps multidimensional boundary corners and per-axis origins' do
          expect_multidimensional_boundaries_to_match(operation, method_name)
        end

        it 'handles periodic folding on an all-odd shape with a larger kernel' do
          signal = NArray[2, 3, 5, 7, 11]
          kernel = NArray[11, 13, 17, 19, 23, 29, 31]
          expect_to_match_reference(
            operation, method_name, signal, kernel, mode: :same, boundary: :wrap, origin: 2
          )
        end
      end
    end
  end

  describe 'the convolution/correlation identity' do
    %i[valid full].each do |mode|
      it "reverses the kernel for #{mode} operations" do
        expect_convolution_identity(mode)
      end
    end

    %i[constant nearest reflect mirror wrap].each do |boundary|
      it "adjusts odd and even origins under #{boundary} boundary extension" do
        expect_same_mode_convolution_identity(boundary)
      end
    end

    it 'adjusts per-axis odd and even origins' do
      expect_per_axis_convolution_identity
    end
  end

  describe 'orientation-sensitive kernels' do
    OperationReference::CALCULATION_METHODS.each do |operation, methods|
      methods.each do |method_name|
        it "orients first, anchored, and last impulses through .#{method_name}" do
          expect_impulse_results(operation, method_name)
        end
      end
    end

    it 'makes convolution and correlation equal for an odd symmetric kernel' do
      expect_symmetric_kernel_results_to_match
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

    OperationReference::ENTRY_POINTS.each do |entry_point|
      invalid_options.each do |options, error|
        it "rejects #{options.inspect} through .#{entry_point}" do
          expect { described_class.public_send(entry_point, validation_signal, validation_kernel, **options) }
            .to raise_error(ArgumentError, error)
        end
      end
    end

    %i[correlate_fft convolve_fft].each do |entry_point|
      it "validates scalar options through .#{entry_point}" do
        expect { described_class.public_send(entry_point, NArray.cast(2), NArray.cast(3), boundary: :reflect) }
          .to raise_error(ArgumentError, /mode: :valid only supports boundary: :constant/)
      end
    end

    %i[same full].each do |mode|
      it "allows large kernels for #{mode} through every entry point" do
        results = OperationReference::ENTRY_POINTS.map do |entry_point|
          described_class.public_send(entry_point, NArray[1, 2], NArray[1, 2, 3], mode:)
        end
        expect(results.length).to eq OperationReference::ENTRY_POINTS.length
      end
    end

    it 'accepts explicitly supplied defaults for valid mode through every entry point' do
      expect_explicit_defaults_to_be_accepted(validation_signal, validation_kernel)
    end
  end

  describe 'cross-implementation coverage' do
    it 'matches independent references across randomized modes, boundaries, and ranks' do
      expect_randomized_cases_to_match_reference
    end

    it 'supports boundary extension on higher-rank inputs' do
      expect_high_rank_boundaries_to_match_reference
    end

    it 'exposes the keyword API from every calculation and estimator method' do
      expected_parameters = [
        %i[req signal], %i[req kernel], %i[key mode], %i[key boundary],
        %i[key fill_value], %i[key origin]
      ]
      parameters = OperationReference::ENTRY_POINTS.map { |method_name| described_class.method(method_name).parameters }
      expect(parameters).to all(eq(expected_parameters))
    end
  end

  describe 'circular FFT selection' do
    it 'uses a real transform when the final axis is even' do
      allow(Numo::Pocketfft).to receive(:rfftn).and_call_original
      described_class.correlate_fft(NArray.ones(3, 4), NArray[1, 2].reshape(1, 2),
                                    mode: :same, boundary: :wrap)
      expect(Numo::Pocketfft).to have_received(:rfftn).at_least(:twice)
    end

    it 'moves a non-final even axis into the real-transform position' do
      exercise_nonfinal_real_axis
      expect(Numo::Pocketfft).to have_received(:rfftn).at_least(:twice)
    end

    it 'preserves results when moving a non-final even axis' do
      result, expected = exercise_nonfinal_real_axis
      expect(result).to be_narray_like expected
    end

    it 'retains the complex transform for all-odd circular shapes' do
      exercise_all_odd_circular_fft
      expect(Numo::Pocketfft).not_to have_received(:rfftn)
    end

    it 'uses a complex transform for all-odd circular shapes' do
      exercise_all_odd_circular_fft
      expect(Numo::Pocketfft).to have_received(:fftn).at_least(:twice)
    end
  end

  describe 'algorithm estimation' do
    it 'produces positive estimates for both operation families' do
      expect(all_estimates).to all(be_positive)
    end

    it 'accounts for the extra correlation spectrum operation' do
      signal = NArray.ones(9)
      kernel = NArray.ones(5)
      expect(described_class.predict_correlate_fft_time(signal, kernel, mode: :same))
        .to be > described_class.predict_convolve_fft_time(signal, kernel, mode: :same)
    end

    it 'estimates periodic transforms below equivalent linear transforms' do
      signal = NArray.ones(9)
      kernel = NArray.ones(5)
      linear = described_class.predict_convolve_fft_time(signal, kernel, mode: :same)
      periodic = described_class.predict_convolve_fft_time(signal, kernel, mode: :same, boundary: :wrap)
      expect(periodic).to be < linear
    end
  end

  describe 'the native valid primitives' do
    it 'keeps both operation-specific methods private' do
      expect(described_class.private_methods).to include(:correlate_basic_valid, :convolve_basic_valid)
    end

    it 'does not expose either primitive publicly' do
      responses = %i[correlate_basic_valid convolve_basic_valid].map do |method_name|
        described_class.respond_to?(method_name)
      end
      expect(responses).to eq [false, false]
    end
  end

  def extent_fixtures(operation)
    return [NArray[13, 26], NArray[5, 13, 26], NArray[5, 13, 26, 12]] if operation == :correlation

    [ConvolutionFixtures.one_dimensional_valid, ConvolutionFixtures.one_dimensional_same,
     ConvolutionFixtures.one_dimensional_full]
  end

  def expect_extent_fixtures_to_match(operation, method_name)
    results = %i[valid same full].map do |mode|
      options = mode == :valid ? {} : { mode: }
      described_class.public_send(method_name, ConvolutionFixtures.one_dimensional_signal,
                                  ConvolutionFixtures.one_dimensional_kernel, **options)
    end
    expect(results.zip(extent_fixtures(operation))).to all(satisfy do |result, fixture|
      result.is_a?(Numo::SFloat) && be_narray_like(fixture).matches?(result)
    end)
  end

  def expect_multidimensional_boundaries_to_match(operation, method_name)
    signal = NArray[[1, 4], [2, 5], [3, 6]].transpose
    kernel = NArray[[1, 3, 5], [2, 4, 6]].transpose
    %i[constant nearest reflect mirror wrap].each do |boundary|
      expect_to_match_reference(
        operation, method_name, signal, kernel, mode: :same, boundary:, origin: [1, -1]
      )
    end
  end

  def expect_convolution_identity(mode)
    signal = NArray[1, 2, 4]
    kernel = NArray[3, 5]
    options = mode == :valid ? {} : { mode: }
    convolution = described_class.convolve(signal, kernel, **options)
    correlation = described_class.correlate(signal, kernel.reverse, **options)
    expect(convolution).to be_narray_like correlation
  end

  def expect_same_mode_convolution_identity(boundary)
    signal = NArray[*Array(1..12)]
    [NArray[2, 3, 5], NArray[2, 3, 5, 7]].each do |kernel|
      (-1..1).each do |origin|
        next unless valid_origin?(kernel.size, origin)

        expect_same_mode_identity(signal, kernel, origin, boundary)
      end
    end
  end

  def expect_same_mode_identity(signal, kernel, origin, boundary)
    correlation_origin = kernel.size.odd? ? -origin : -origin - 1
    options = { mode: :same, boundary:, origin: }
    convolution = described_class.convolve(signal, kernel, **options)
    correlation = described_class.correlate(signal, kernel.reverse, **options, origin: correlation_origin)
    expect(convolution).to be_narray_like correlation
  end

  def expect_per_axis_convolution_identity
    signal = NArray[*Array(1..20)].reshape(4, 5)
    kernel = NArray[*Array(1..6)].reshape(3, 2)
    options = { mode: :same, boundary: :mirror, origin: [1, -1] }
    convolution = described_class.convolve(signal, kernel, **options)
    correlation = described_class.correlate(signal, kernel.reverse(0, 1), **options, origin: [-1, 0])
    expect(convolution).to be_narray_like correlation
  end

  def expect_impulse_results(operation, method_name)
    signal = NArray[1, 2, 4, 8]
    results = impulse_kernels.map { |kernel| described_class.public_send(method_name, signal, kernel) }
    expect(results.zip(impulse_expectations(operation))).to all(satisfy do |result, value|
      be_narray_like(value).matches?(result)
    end)
  end

  def impulse_kernels
    [NArray[1, 0, 0], NArray[0, 1, 0], NArray[0, 0, 1]]
  end

  def impulse_expectations(operation)
    correlation = [NArray[1, 2], NArray[2, 4], NArray[4, 8]]
    operation == :correlation ? correlation : correlation.reverse
  end

  def expect_symmetric_kernel_results_to_match
    signal = NArray[*Array(1..12)].reshape(3, 4)
    kernel = NArray[[1, 2, 1], [3, 5, 3], [1, 2, 1]]
    pairs = OperationReference::CALCULATION_METHODS.values.transpose.map do |methods|
      calculate_methods(methods, signal, kernel)
    end
    expect(pairs).to all(satisfy { |correlation, convolution| be_narray_like(correlation).matches?(convolution) })
  end

  def calculate_methods(methods, signal, kernel)
    methods.map { |method_name| described_class.public_send(method_name, signal, kernel) }
  end

  def expect_explicit_defaults_to_be_accepted(signal, kernel)
    results = OperationReference::ENTRY_POINTS.map do |entry_point|
      described_class.public_send(
        entry_point, signal, kernel, mode: :valid, boundary: :constant, fill_value: 0, origin: [0]
      )
    end
    expect(results.length).to eq OperationReference::ENTRY_POINTS.length
  end

  def expect_randomized_cases_to_match_reference
    random = Random.new(12_345)
    OperationReference::OPERATIONS.each do |operation|
      randomized_cases.each do |case_data|
        expect_randomized_case_to_match_reference(operation, case_data, random)
      end
    end
  end

  def expect_high_rank_boundaries_to_match_reference
    signal = NArray[*Array(1..8)].reshape(2, 2, 2)
    kernel = NArray[*Array(1..12)].reshape(3, 2, 2)
    options = { mode: :same, boundary: :mirror, origin: [1, -1, 0] }
    OperationReference::OPERATIONS.each do |operation|
      expected = OperationReference.calculate(operation, signal, kernel, **options)
      OperationReference::CALCULATION_METHODS.fetch(operation).each do |method_name|
        expect(described_class.public_send(method_name, signal, kernel, **options)).to be_narray_like expected
      end
    end
  end

  def exercise_nonfinal_real_axis
    signal = NArray[*Array(1..12)].reshape(4, 3)
    kernel = NArray[[1, 2], [3, 5], [7, 11]]
    allow(Numo::Pocketfft).to receive(:rfftn).and_call_original
    options = { mode: :same, boundary: :wrap }
    expected = OperationReference.calculate(:convolution, signal, kernel, **options)
    [described_class.convolve_fft(signal, kernel, **options), expected]
  end

  def exercise_all_odd_circular_fft
    allow(Numo::Pocketfft).to receive(:rfftn).and_call_original
    allow(Numo::Pocketfft).to receive(:fftn).and_call_original
    described_class.correlate_fft(
      NArray.ones(3, 5), NArray.ones(3, 3), mode: :same, boundary: :wrap
    )
  end

  def all_estimates
    signal = NArray.ones(9)
    kernel = NArray.ones(5)
    OperationReference::ESTIMATOR_METHODS.values.flatten.map do |method_name|
      described_class.public_send(method_name, signal, kernel, mode: :same)
    end
  end

  def expect_cases_to_match_reference(operation, method_name, cases)
    cases.each do |signal, kernel, options|
      expect_to_match_reference(operation, method_name, signal, kernel, options)
    end
  end

  def expect_to_match_reference(operation, method_name, signal, kernel, options = {})
    reference_options = { mode: :valid, **options }
    expected = OperationReference.calculate(operation, signal, kernel, **reference_options)
    expect(described_class.public_send(method_name, signal, kernel, **options)).to be_narray_like expected
  end

  def valid_origin?(kernel_size, origin)
    ((kernel_size / 2) + origin).between?(0, kernel_size - 1)
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

  def expect_randomized_case_to_match_reference(operation, case_data, random)
    signal, kernel = random_arrays(case_data.first(2), random)
    options = case_options(case_data.drop(2))
    expected = OperationReference.calculate(operation, signal, kernel, **options, fill_value: case_data[4])
    results = OperationReference::CALCULATION_METHODS.fetch(operation).drop(1).map do |method_name|
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
end
