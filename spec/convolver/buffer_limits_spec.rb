# frozen_string_literal: true

require 'helpers'

module Convolver
  describe BufferLimits do
    [32, 64].each do |bits|
      context "with #{bits}-bit sizes" do
        subject(:limits) { described_class.new(size_max:, byte_max: size_max / 4) }

        let(:size_max) { (1 << bits) - 1 }

        [4, 8, 16].each do |width|
          it "accepts the last representable #{width}-byte buffer" do
            expect { limits.buffer!(limits.byte_max / width, width, 'data') }.not_to raise_error
          end

          it "rejects the next #{width}-byte buffer without multiplying" do
            expect { limits.buffer!((limits.byte_max / width) + 1, width, 'data') }
              .to raise_error(RangeError, 'data exceeds native byte/stride limit')
          end
        end

        it 'accepts the element product at SIZE_MAX' do
          expect(limits.product!([1, size_max], 'data')).to eq size_max
        end

        it 'rejects an overflowing element product' do
          expect { limits.product!([size_max, 2], 'data') }
            .to raise_error(RangeError, 'data exceeds native element limit')
        end
      end
    end

    it 'uses the positive signed tagged-stride ceiling' do
      expect(described_class::BYTE_MAX).to eq((described_class::SIZE_MAX / 2) / 2)
    end

    it 'counts a scalar as one element' do
      expect(described_class::NATIVE.product!([], 'scalar')).to eq 1
    end
  end
end
