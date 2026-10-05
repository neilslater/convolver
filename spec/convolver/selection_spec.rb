# frozen_string_literal: true

require 'helpers'

module Convolver
  describe Convolver do
    before { allow(BufferPreparation).to receive(:new).and_call_original }

    %i[convolve correlate].each do |method|
      [999, 1000, 1001, 4096].each do |size|
        it "skips FFT planning for a cheap #{method} at size #{size}", :aggregate_failures do
          allow(FftBufferPlan).to receive(:new).and_call_original
          result = described_class.public_send(method, Numo::SFloat.ones(size), Numo::SFloat.ones(3))
          expect(result).to be_narray_like Numo::SFloat.ones(size - 2) * 3
          expect(FftBufferPlan).not_to have_received(:new)
          expect(BufferPreparation).to have_received(:new).once
        end
      end

      it "uses FFT for a worthwhile #{method} below the old size threshold", :aggregate_failures do
        allow(Numo::Pocketfft).to receive(:rfftn).and_call_original
        result = described_class.public_send(method, Numo::SFloat.ones(999), Numo::SFloat.ones(500))
        expect(result).to be_narray_like Numo::SFloat.ones(500) * 500
        expect(Numo::Pocketfft).to have_received(:rfftn).twice
      end

      it "uses the transform lower bound to avoid planning #{method}", :aggregate_failures do
        allow(FftBufferPlan).to receive(:new).and_call_original
        result = described_class.public_send(method, Numo::SFloat.ones(64, 64), Numo::SFloat.ones(5, 5))
        expect(result).to be_narray_like Numo::SFloat.ones(60, 60) * 25
        expect(FftBufferPlan).not_to have_received(:new)
      end
    end
  end
end
