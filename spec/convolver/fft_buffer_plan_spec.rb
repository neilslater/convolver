# frozen_string_literal: true

require 'helpers'

module Convolver
  describe FftBufferPlan do
    def plan_for(signal_shape, kernel_shape, mode: :valid, boundary: :constant)
      OperationPlan.new(Numo::SFloat.ones(*signal_shape), Numo::SFloat.ones(*kernel_shape),
                        operation: :convolution, mode:, boundary:, fill_value: UNSPECIFIED_FILL, origin: 0)
    end

    it 'retains the exact shape when only optional padding exceeds the byte limit' do
      plan = plan_for([11, 14], [1, 1])
      buffers = described_class.new(plan, [11, 14], [1, 1], limits: BufferLimits.new(byte_max: 2464))

      expect(buffers.transform_shape).to eq [11, 14]
    end

    it 'rejects all candidates when inverse staging cannot fit' do
      plan = plan_for([11, 14], [1, 1])
      expect { described_class.new(plan, [11, 14], [1, 1], limits: BufferLimits.new(byte_max: 2463)) }
        .to raise_error(FftUnavailable, /FFT inverse staging buffer/)
    end

    it 'checks an original periodic kernel larger than the signal' do
      plan = plan_for([4], [200], mode: :same, boundary: :wrap)
      expect { described_class.new(plan, [4], [200], limits: BufferLimits.new(byte_max: 1200)) }
        .to raise_error(FftUnavailable, /circular kernel preparation/)
    end

    it 'preserves the periodic shape and moved real axis', :aggregate_failures do
      plan = plan_for([4, 3], [2, 2], mode: :same, boundary: :wrap)

      expect(plan.fft_buffers.transform_shape).to eq [4, 3]
      expect(plan.fft_buffers.real_axis).to eq 0
      expect(plan.fft_buffers.spectrum_size).to eq 9
    end

    it 'preserves all-odd periodic dimensions', :aggregate_failures do
      plan = plan_for([3, 5], [2, 2], mode: :same, boundary: :wrap)

      expect(plan.fft_buffers.transform_shape).to eq [3, 5]
      expect(plan.fft_buffers.real_axis).to be_nil
      expect(plan.fft_buffers.spectrum_size).to eq 15
    end

    it 'reuses the same frozen transform plan', :aggregate_failures do
      plan = plan_for([11, 14], [1, 1])

      expect(plan.fft_buffers).to equal plan.fft_buffers
      expect(plan.fft_buffers.transform_shape).to be_frozen
    end
  end
end
