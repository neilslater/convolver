# frozen_string_literal: true

require 'helpers'

describe Convolver do
  %i[convolve_fft correlate_fft].each do |method_name|
    %i[valid same full].each do |mode|
      it "multiplies scalar operands through .#{method_name} in #{mode} mode" do
        result = described_class.public_send(method_name, NArray.cast(-2), NArray.cast(3), mode:)
        expect(result).to be_narray_like NArray.cast(-6)
      end
    end
  end

  { predict_convolve_fft_time: 0, predict_correlate_fft_time: 2e-10 }.each do |method_name, correlation_cost|
    it "estimates a scalar linear transform as one element through .#{method_name}" do
      scalar = described_class.public_send(method_name, NArray.cast(2), NArray.cast(3))
      expected = 1.5e-4 + (1.45e-9 * Math.log(2)) + 2e-9 + correlation_cost
      expect(scalar).to be_within(1e-15).of(expected)
    end

    it "estimates a scalar circular transform through .#{method_name}" do
      scalar = described_class.public_send(method_name, NArray.cast(2), NArray.cast(3),
                                           mode: :same, boundary: :wrap)
      expected = 1.5e-4 + (2.8e-9 * Math.log(2)) + 2e-9 + correlation_cost
      expect(scalar).to be_within(1e-15).of(expected)
    end
  end
end
