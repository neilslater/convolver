# frozen_string_literal: true

require 'helpers'

module Convolver
  describe BufferPreparation do
    let(:limits) { BufferLimits.new(byte_max: 64) }
    let(:plan) do
      instance_double(OperationPlan, result_size: 1, extended_size: 16, extended_shape: [16],
                                     valid?: false, wrap?: false, boundary: :nearest)
    end

    it 'checks boundary index bytes separately from numeric extension bytes' do
      expect { described_class.new(plan, [16], [1], limits:).basic! }
        .to raise_error(RangeError, /boundary index buffer/)
    end

    it 'checks signal conversion even when the output is small' do
      expect { described_class.new(plan, [17], [1], limits:).basic! }
        .to raise_error(RangeError, /signal conversion buffer/)
    end

    it 'checks kernel conversion independently from signal and output' do
      expect { described_class.new(plan, [1], [17], limits:).basic! }
        .to raise_error(RangeError, /kernel conversion buffer/)
    end

    it 'checks circular signal casts before constructing the operation' do
      allow(plan).to receive(:wrap?).and_return(true)

      expect { described_class.new(plan, [10], [1], limits:).fft! }
        .to raise_error(FftUnavailable, /circular signal preparation/)
    end
  end
end
