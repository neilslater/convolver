# frozen_string_literal: true

require 'helpers'

module Convolver
  describe Convolver do
    [CostProfiles::MAC, CostProfiles::X86, CostProfiles::ARM, nil].each_with_index do |profile, index|
      context "with profile #{index}" do
        before { stub_const('Convolver::CostProfiles::CURRENT', profile) }

        [Numo::SFloat, Numo::DFloat].each do |type|
          it "retains the rank-four short-kernel estimate for #{type}" do
            rate = type == Numo::DFloat ? 4.6e-10 : 4.5e-10
            estimate = described_class.predict_convolve_basic_time(type.ones(8, 8, 8, 8), type.ones(2, 2, 2, 2))
            expect(estimate).to be_within(1e-15).of(rate * (7**4) * 16)
          end

          it "retains the rank-four long-kernel estimate for #{type}" do
            estimate = described_class.predict_correlate_basic_time(type.ones(8, 8, 8, 8), type.ones(3, 3, 3, 3))
            expect(estimate).to be_within(1e-15).of(6.5e-10 * (6**4) * 81)
          end
        end
      end
    end

    context 'with an uncalibrated platform' do
      before { stub_const('Convolver::CostProfiles::CURRENT', nil) }

      { Numo::SFloat => [3.8e-10, 5.9e-10], Numo::DFloat => [3.9e-10, 6e-10] }.each do |type, rates|
        it "retains the rank-one short-kernel estimate for #{type}" do
          estimate = described_class.predict_convolve_basic_time(type.ones(999), type.ones(3))
          expect(estimate).to be_within(1e-15).of(rates.first * 997 * 3)
        end

        it "retains the rank-two estimate for #{type}" do
          estimate = described_class.predict_correlate_basic_time(type.ones(8, 8), type.ones(3, 3))
          expect(estimate).to be_within(1e-15).of(rates.last * 36 * 9)
        end
      end

      it 'retains the 0.8 margin for a marginal predicted FFT gain', :aggregate_failures do
        inputs = [Numo::SFloat.ones(999), Numo::SFloat.ones(500)]
        stub_marginal_fft(*inputs)
        result = described_class.convolve(*inputs)
        expect(result).to be_narray_like Numo::SFloat.ones(500) * 500
        expect(Numo::Pocketfft).not_to have_received(:rfftn)
      end
    end

    def stub_marginal_fft(signal, kernel)
      estimate = 0.85 * described_class.predict_convolve_basic_time(signal, kernel)
      allow(FftEstimator).to receive(:new).and_return(instance_double(FftEstimator, call: estimate))
      allow(Numo::Pocketfft).to receive(:rfftn).and_call_original
    end
  end
end
