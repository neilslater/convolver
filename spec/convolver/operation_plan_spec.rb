# frozen_string_literal: true

require 'helpers'

module Convolver
  describe OperationPlan do
    let(:plan) do
      described_class.new(Numo::SFloat.ones(10), Numo::SFloat.ones(3),
                          operation: :convolution, mode: :valid, boundary: :constant,
                          fill_value: UNSPECIFIED_FILL, origin: 0)
    end

    it 'memoizes successful direct validation' do
      allow(BufferPreparation).to receive(:new).and_call_original
      2.times { plan.validate_basic! }
      expect(BufferPreparation).to have_received(:new).once
    end

    it 'does not memoize a failed validation' do
      preparation = instance_double(BufferPreparation)
      allow(BufferPreparation).to receive(:new).and_return(preparation)
      allow(preparation).to receive(:basic!).and_raise(RangeError, 'buffer limit')
      2.times { expect { plan.validate_basic! }.to raise_error(RangeError, 'buffer limit') }
    end
  end
end
