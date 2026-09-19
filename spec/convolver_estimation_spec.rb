# frozen_string_literal: true

require 'helpers'

describe Convolver do
  describe 'circular FFT cost estimates' do
    it 'accounts for a real transform on the final even axis' do
      # Twelve spatial elements, one kernel element, and no axis movement.
      expected = 1.5e-4 + (9e-10 * 12 * Math.log(12)) + (1e-9 * 13)
      expect(circular_estimate(:predict_convolve_fft_time, [3, 4])).to be_within(1e-15).of(expected)
    end

    it 'charges for moving both inputs and the result when the even axis is non-final' do
      final_axis = circular_estimate(:predict_convolve_fft_time, [3, 4])
      moved_axis = circular_estimate(:predict_convolve_fft_time, [4, 3])
      expect(moved_axis - final_axis).to be_within(1e-15).of(3 * 12 * 1.6e-9)
    end

    [[3, 4], [4, 3]].each do |shape|
      it "charges correlation for nine stored real-spectrum elements on #{shape}" do
        correlation = circular_estimate(:predict_correlate_fft_time, shape)
        convolution = circular_estimate(:predict_convolve_fft_time, shape)
        expect(correlation - convolution).to be_within(1e-15).of(9 * 2e-10)
      end
    end
  end

  def circular_estimate(method_name, shape)
    described_class.public_send(method_name, NArray.ones(*shape), NArray.ones(1, 1),
                                mode: :same, boundary: :wrap)
  end
end
