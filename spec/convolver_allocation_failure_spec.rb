# frozen_string_literal: true

require 'helpers'

module Convolver
  describe Convolver do
    let(:signal) { NArray.ones(4096) }
    let(:estimator) { instance_double(FftEstimator, call: 0.0) }
    let(:kernel) { NArray.ones(256) }

    %i[convolve correlate].each do |operation|
      context "with #{operation} allocation exhaustion" do
        before do
          allow(FftEstimator).to receive(:new).and_return(estimator)
          allow(Numo::DFloat).to receive(:zeros).and_raise(NoMemoryError, 'simulated allocator exhaustion')
          allow(described_class).to receive("#{operation}_basic_valid").and_call_original
        end

        it 'propagates NoMemoryError without an automatic direct retry', :aggregate_failures do
          expect { described_class.public_send(operation, signal, kernel) }
            .to raise_error(NoMemoryError, 'simulated allocator exhaustion')
          expect(described_class).not_to have_received("#{operation}_basic_valid")
          expect(Numo::DFloat).to have_received(:zeros).once
        end
      end

      it "propagates arbitrary estimator RangeError through #{operation}" do
        allow(FftEstimator).to receive(:new).and_raise(RangeError, 'unrelated error')

        expect { described_class.public_send(operation, signal, kernel) }.to raise_error(RangeError, 'unrelated error')
      end
    end
  end
end
