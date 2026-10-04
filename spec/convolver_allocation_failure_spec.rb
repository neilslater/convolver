# frozen_string_literal: true

require 'helpers'

describe Convolver do
  let(:signal) { NArray.ones(1000) }
  let(:kernel) { NArray.ones(3) }

  %i[convolve correlate].each do |operation|
    context "with #{operation} allocation exhaustion" do
      before do
        allow(described_class).to receive("predict_#{operation}_fft_time").and_return(0.0)
        allow(Numo::DFloat).to receive(:zeros).and_raise(NoMemoryError, 'simulated allocator exhaustion')
        allow(described_class).to receive("#{operation}_basic").and_call_original
      end

      it 'propagates NoMemoryError without an automatic direct retry', :aggregate_failures do
        expect { described_class.public_send(operation, signal, kernel) }
          .to raise_error(NoMemoryError, 'simulated allocator exhaustion')
        expect(described_class).not_to have_received("#{operation}_basic")
        expect(Numo::DFloat).to have_received(:zeros).once
      end
    end

    it "propagates arbitrary estimator RangeError through #{operation}" do
      allow(described_class).to receive("predict_#{operation}_fft_time").and_raise(RangeError, 'unrelated error')

      expect { described_class.public_send(operation, signal, kernel) }.to raise_error(RangeError, 'unrelated error')
    end
  end
end
