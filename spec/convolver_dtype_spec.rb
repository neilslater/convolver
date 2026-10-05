# frozen_string_literal: true

require 'helpers'
require 'dtype_fixtures'

describe Convolver do
  include DtypeFixtures

  OperationReference::CALCULATION_METHODS.each do |operation, methods|
    methods.each do |method_name|
      DtypeFixtures::INPUTS.keys.product(DtypeFixtures::INPUTS.keys).each do |first, second|
        it "promotes #{first}/#{second} through #{method_name}", :aggregate_failures do
          expected = operation == :convolution ? [5, 8] : [4, 7]
          DtypeFixtures::OPTIONS.each do |options|
            result = described_class.public_send(method_name, first[1, 2, 3], second[2, 1], **options)
            expect_dtype_result(result, expected_dtype(first, second, options), expected)
          end
        end
      end

      it "retains cancellation detail through #{method_name}", :aggregate_failures do
        expect_cancellation_precision(method_name)
      end

      DtypeFixtures::MODES.each do |options|
        it "preserves double precision in #{method_name} #{options}", :aggregate_failures do
          signal = Numo::DFloat[[1, 2, 3], [4, 5, 6]] + (2.0**-30)
          kernel = Numo::SFloat[[0.5, -1], [1, 0.25]]
          result = described_class.public_send(method_name, signal, kernel, **options)
          expected = OperationReference.calculate(operation, signal, kernel, **options)
          expect_dtype_result(result, Numo::DFloat, expected)
        end
      end

      it "casts inputs before periodic fold collisions in #{method_name}", :aggregate_failures do
        expect_periodic_precision(method_name)
      end

      it "preserves scalar precision in #{method_name}" do
        result = described_class.public_send(method_name, Numo::DFloat.cast(1 + (2.0**-30)), Numo::Int8.cast(2))
        expect(result.to_f).to eq(2 + (2.0**-29))
      end
    end
  end

  OperationReference::ENTRY_POINTS.each do |method_name|
    [Numo::Int32, Numo::DComplex, :sfloat, 'DFloat', false, Class.new(Numo::DFloat)].each do |dtype|
      it "rejects explicit #{dtype.inspect} in #{method_name}" do
        expect { described_class.public_send(method_name, NArray[1], NArray[1], dtype:) }
          .to raise_error(ArgumentError, /dtype must be/)
      end
    end

    [Numo::Bit, Numo::RObject, Numo::SComplex, Numo::DComplex, Class.new(Numo::SFloat)].each do |type|
      it "rejects #{type} inputs despite an override in #{method_name}" do
        expect { described_class.public_send(method_name, NArray[1], type.new(1).fill(1), dtype: Numo::SFloat) }
          .to raise_error(ArgumentError, /unsupported signal or kernel dtype/)
      end
    end
  end
  def expect_cancellation_precision(method_name)
    signal = Numo::DFloat[16_777_217, -16_777_216]
    kernel = Numo::SFloat[1, 1]
    [[signal, kernel], [kernel, signal]].each do |inputs|
      expect_dtype_result(described_class.public_send(method_name, *inputs), Numo::DFloat, [1])
      expect_dtype_result(described_class.public_send(method_name, *inputs, dtype: Numo::SFloat), Numo::SFloat, [0])
    end
  end

  def expect_periodic_precision(method_name)
    signal = Numo::DFloat[1, 2]
    kernel = Numo::DFloat[16_777_217, 0, -16_777_216, 0]
    result = described_class.public_send(method_name, signal, kernel, mode: :same, boundary: :wrap)
    expect_dtype_result(result, Numo::DFloat, [1, 2])
    narrowed = described_class.public_send(method_name, signal, kernel, mode: :same, boundary: :wrap,
                                                                        dtype: Numo::SFloat)
    expect_dtype_result(narrowed, Numo::SFloat, [0, 0])
  end
end
