# frozen_string_literal: true

require 'helpers'

module Convolver
  describe OperationShapes do
    it 'rejects overflow of an output dimension before allocation' do
      expect do
        described_class.new([described_class::SIZE_MAX], [2],
                            operation: :convolution, mode: :full, anchors: [1])
      end.to raise_error(RangeError, 'result shape exceeds native implementation limit')
    end

    it 'rejects overflow of the output element count before allocation' do
      expect do
        described_class.new([described_class::SIZE_MAX, 2], [1, 1],
                            operation: :convolution, mode: :valid, anchors: [0, 0])
      end.to raise_error(RangeError, 'result size exceeds native implementation limit')
    end
  end
end
