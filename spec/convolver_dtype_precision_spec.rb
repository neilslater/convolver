# frozen_string_literal: true

require 'helpers'

describe Convolver do
  OperationReference::CALCULATION_METHODS.values.flatten.each do |method_name|
    [Numo::SFloat, Numo::DFloat].each do |dtype|
      it "casts fill without promoting #{dtype} in #{method_name}" do
        expected = dtype[16_777_217, 3, 16_777_218]
        result = described_class.public_send(method_name, dtype[1, 2], dtype[1, 1],
                                             mode: :full, fill_value: 16_777_216, dtype:)
        expect(result).to be_narray_like(expected, 1e-10)
      end

      it "keeps the #{dtype} result with Ruby Float fill in #{method_name}" do
        result = described_class.public_send(method_name, dtype[1], dtype[1, 1], mode: :full, fill_value: 0.1)
        expect(result).to be_an_instance_of(dtype)
      end
    end

    it "casts fill to SFloat before calculation in #{method_name}" do
      result = described_class.public_send(method_name, Numo::DFloat[0], Numo::DFloat[1, 1],
                                           mode: :full, fill_value: 16_777_217, dtype: Numo::SFloat)
      expect(result.to_a).to eq [16_777_216.0, 16_777_216.0]
    end

    it "preserves fractional DFloat products in #{method_name}" do
      result = described_class.public_send(method_name, Numo::DFloat[1 + (2.0**-30)], Numo::DFloat[1 + (2.0**-30)])
      expect(result.to_f).to be_within(1e-15).of((1 + (2.0**-30))**2)
    end
  end

  %i[convolve_basic correlate_basic].each do |method_name|
    it "uses double products and accumulation in #{method_name}" do
      kernel = method_name == :convolve_basic ? [4097, -4098] : [-4098, 4097]
      result = described_class.public_send(method_name, Numo::DFloat[4096, 4097], Numo::DFloat.cast(kernel))
      expect(result.to_a).to eq [1.0]
    end
  end

  %i[convolve_fft correlate_fft].each do |method_name|
    it "rounds scalar overflow and underflow normally in #{method_name}", :aggregate_failures do
      expect(scalar_product(method_name, 1e300)).to eq Float::INFINITY
      expect(scalar_product(method_name, 1e-50)).to eq 0.0
      expect(scalar_product(method_name, Float::NAN)).to be_nan
    end
  end

  def scalar_product(method_name, value)
    described_class.public_send(method_name, Numo::DFloat.cast(value), Numo::SFloat.cast(1), dtype: Numo::SFloat).to_f
  end
end
