# frozen_string_literal: true

require 'helpers'

module Convolver
  describe Convolver do
    { mac: CostProfiles::MAC, x86: CostProfiles::X86, arm: CostProfiles::ARM }.each do |name, profile|
      context "with the #{name} profile" do
        before { stub_const('Convolver::CostProfiles::CURRENT', profile) }

        %i[convolve correlate].product([Numo::SFloat, Numo::DFloat]).each do |method, type|
          context "with #{type} #{method}" do
            it 'keeps the 769/65 crossover on the direct path' do
              expect_selection(method, type, [769], [65], fft: false)
            end

            it 'preserves worthwhile large 3D FFT execution' do
              expect_selection(method, type, [32, 32, 32], [15, 15, 15], fft: true)
            end

            it 'accounts for platform and dtype in the 2D crossover' do
              expect_selection(method, type, [64, 64], [16, 16], fft: name != :x86 || type == Numo::DFloat)
            end

            it 'accounts for the cost of periodic folding' do
              expect_selection(method, type, [32, 32], [48, 48], fft: name == :mac, mode: :same, boundary: :wrap)
            end

            it 'avoids a sudden estimate jump above a 64-element kernel' do
              estimates = [63, 64, 65].map do |size|
                described_class.public_send("predict_#{method}_basic_time", type.ones(999 + size), type.ones(size))
              end
              expect(estimates.last / estimates[1]).to be_between(1.0, 1.05)
            end
          end
        end
      end
    end

    def expect_selection(method, type, signal_shape, kernel_shape, fft:, **options)
      allow(Numo::Pocketfft).to receive(:rfftn).and_call_original
      signal, kernel = [signal_shape, kernel_shape].map { |shape| type.ones(*shape) }
      result = described_class.public_send(method, signal, kernel, **options)
      expect_constant_result(result, type, kernel.size)
      expect(Numo::Pocketfft).to have_received(:rfftn).exactly(fft ? 2 : 0).times
    end

    def expect_constant_result(result, type, value)
      expect(result).to be_narray_like type.ones(*result.shape) * value
      expect(result).to be_an_instance_of(type)
    end
  end
end
