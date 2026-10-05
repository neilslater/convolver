# frozen_string_literal: true

require 'helpers'
require 'buffer_validation_helpers'

module Convolver
  describe Convolver do
    include BufferValidationHelpers

    let(:signal) { NArray.ones(1000) }
    let(:kernel) { NArray.ones(3) }
    let(:periodic_inputs) { [NArray.ones(4), NArray.ones(200)] }

    %i[convolve correlate].each do |operation|
      context "with #{operation} buffer validation" do
        before do
          inject_limits(byte_max: 10_000)
          watch_allocations
        end

        [%i[valid constant], %i[same constant], %i[full constant],
         %i[same nearest], %i[same wrap]].each do |mode, boundary|
          it "rejects #{mode}/#{boundary} execution before preparation" do
            expect { described_class.public_send("#{operation}_fft", signal, kernel, mode:, boundary:) }
              .to raise_error(FftUnavailable, /FFT inverse staging buffer/)
            expect_no_allocations
          end

          it "rejects the same #{mode}/#{boundary} FFT estimator before preparation" do
            expect { described_class.public_send("predict_#{operation}_fft_time", signal, kernel, mode:, boundary:) }
              .to raise_error(FftUnavailable, /FFT inverse staging buffer/)
            expect_no_allocations
          end
        end

        it 'automatically falls back to safe direct execution', :aggregate_failures do
          expected = OperationReference.calculate(operation == :convolve ? :convolution : :correlation,
                                                  signal, kernel, mode: :valid)
          expect(described_class.public_send(operation, signal, kernel)).to be_narray_like expected
          expect(Numo::Pocketfft).not_to have_received(:rfftn)
        end

        it 'rejects an oversized original periodic kernel before casting or folding' do
          inject_limits(byte_max: 1200)
          expect { described_class.public_send("#{operation}_fft", *periodic_inputs, mode: :same, boundary: :wrap) }
            .to raise_error(FftUnavailable, /circular kernel preparation/)
          expect_no_allocations
        end

        it 'does not swallow ordinary validation errors in automatic selection' do
          expect { described_class.public_send(operation, signal, kernel, mode: :full, boundary: :wrap) }
            .to raise_error(ArgumentError, /only supports boundary/)
          expect_no_allocations
        end
      end

      context "with #{operation} common buffer limits" do
        before { watch_allocations }

        it 'checks the resolved double width before casting either input' do
          inject_limits(byte_max: 7999)
          expect { described_class.public_send(operation, signal, kernel, dtype: Numo::DFloat, mode: :same) }
            .to raise_error(RangeError, /result buffer/)
          expect_no_allocations
        end

        %w[basic fft].each do |algorithm|
          it "checks extension before #{algorithm} preparation" do
            inject_limits(byte_max: 4000)
            expect { described_class.public_send("#{operation}_#{algorithm}", signal, kernel, mode: :same) }
              .to raise_error(RangeError, /extended signal buffer/)
            expect_no_allocations
          end

          it "checks the result buffer before #{algorithm} preparation" do
            inject_limits(byte_max: 3999)
            expect { described_class.public_send("#{operation}_#{algorithm}", signal, kernel, mode: :same) }
              .to raise_error(RangeError, /result buffer/)
            expect_no_allocations
          end
        end

        it 'requires the fallback direct preparation to be representable too' do
          inject_limits(byte_max: 4000)
          expect { described_class.public_send(operation, signal, kernel, mode: :same) }
            .to raise_error(RangeError, /extended signal buffer/)
          expect_no_allocations
        end

        it 'rejects direct kernel-cache byte overflow before native entry' do
          inject_limits(byte_max: 4000)
          expect { described_class.public_send("#{operation}_basic", signal, NArray.ones(501)) }
            .to raise_error(RangeError, /direct kernel offset cache/)
          expect_no_allocations
        end
      end

      context "with #{operation} native integer limits" do
        before { inject_limits(int_max: 1600) }

        it 'agrees between FFT execution and estimation', :aggregate_failures do
          expect { described_class.public_send("#{operation}_fft", signal, kernel) }
            .to raise_error(FftUnavailable, /FFT axis arithmetic/)
          expect { described_class.public_send("predict_#{operation}_fft_time", signal, kernel) }
            .to raise_error(FftUnavailable, /FFT axis arithmetic/)
        end

        it 'does not impose FFT integer limits on direct execution' do
          expect(described_class.public_send("#{operation}_basic", signal, kernel).shape).to eq [998]
        end

        it 'skips FFT-only limits for scalar execution and estimation', :aggregate_failures do
          inject_limits(byte_max: 8, int_max: 0)
          expect(described_class.public_send("#{operation}_fft", NArray.cast(2), NArray.cast(3)).to_f).to eq 6
          expect(described_class.public_send("predict_#{operation}_fft_time", NArray.cast(2), NArray.cast(3),
                                             mode: :same, boundary: :wrap)).to be_positive
        end
      end
    end
  end
end
