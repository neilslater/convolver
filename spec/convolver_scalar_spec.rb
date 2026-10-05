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

  %i[predict_convolve_fft_time predict_correlate_fft_time].each do |method_name|
    it "estimates the scalar shortcut below a full transform through .#{method_name}" do
      scalar = described_class.public_send(method_name, NArray.cast(2), NArray.cast(3))
      transform = described_class.public_send(method_name, NArray.ones(1024), NArray.ones(3))
      expect(scalar).to be_between(0, transform).exclusive
    end

    it "does not charge scalar calls for periodic folding through .#{method_name}" do
      linear = described_class.public_send(method_name, NArray.cast(2), NArray.cast(3))
      scalar = described_class.public_send(method_name, NArray.cast(2), NArray.cast(3),
                                           mode: :same, boundary: :wrap)
      expect(scalar).to eq(linear)
    end
  end
end
