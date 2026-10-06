# frozen_string_literal: true

require 'helpers'
require_relative 'prototype'

# Deterministic checks for the isolated research implementation.
module Recalibration
  describe Prototype do
    let(:signal) { Numo::SFloat.ones(999) }
    let(:kernel) { Numo::SFloat.ones(500) }
    let(:execution) { described_class.new(:convolution, signal, kernel, **options) }

    [Profiles::MAC, Profiles::X86, Profiles::ARM].each_with_index do |profile, index|
      context "with profile #{index}" do
        let(:plan) { Convolver.const_get(:OperationPlan).new(signal, kernel, operation: :convolution, **options) }
        let(:cost_model) { CostModel.new(:convolution, signal, kernel, plan, profile) }

        before { stub_const('Recalibration::Profiles::CURRENT', profile) }

        it 'bounds seeded transforms', :aggregate_failures do
          random = Random.new(62_418)
          100.times do
            lower, full = random_bounds(random, profile)
            expect(lower).to be <= full
          end
        end

        it 'does not construct FFT buffers for the bound' do
          allow(plan).to receive(:fft_buffers).and_call_original
          cost_model.lower_bound(Float::INFINITY)
          expect(plan).not_to have_received(:fft_buffers)
        end

        it 'keeps overflow-sized metadata representable in the bound' do
          plan = Convolver.const_get(:OperationPlan).new(signal, kernel, operation: :convolution, **options)
          allow(plan).to receive(:extended_shape).and_return([Convolver.const_get(:BufferLimits)::SIZE_MAX])
          model = CostModel.new(:convolution, signal, kernel, plan, profile)
          expect(model.lower_bound(Float::INFINITY)).to be_finite
        end

        it 'preflights direct storage once and preserves the result', :aggregate_failures do
          allow(Convolver.const_get(:BufferPreparation)).to receive(:new).and_call_original
          actual = described_class.call(:convolution, Numo::SFloat.ones(999), Numo::SFloat.ones(3))
          expect(actual).to be_narray_like Numo::SFloat.ones(997) * 3
          expect(Convolver.const_get(:BufferPreparation)).to have_received(:new).once
        end

        it 'estimates without converting input data', :aggregate_failures do
          allow(Numo::SFloat).to receive(:cast).and_call_original
          allow(Numo::DFloat).to receive(:cast).and_call_original
          expect([cost_model.basic_time, cost_model.fft_time]).to all(be_positive)
          expect(Numo::SFloat).not_to have_received(:cast)
          expect(Numo::DFloat).not_to have_received(:cast)
        end
      end
    end

    context 'with an unavailable FFT' do
      let(:model) { instance_double(CostModel, basic_time: 1.0, lower_bound: 0.0) }

      before do
        stub_const('Recalibration::Profiles::CURRENT', Profiles::MAC)
        allow(execution).to receive(:cost_model).and_return(model)
        allow(execution).to receive(:basic).and_call_original
      end

      it 'falls back only on FftUnavailable', :aggregate_failures do
        allow(model).to receive(:fft_time).and_raise(Convolver.const_get(:FftUnavailable))
        expect(execution.automatic).to be_narray_like Numo::SFloat.ones(500) * 500
        expect(execution).to have_received(:basic).once
      end

      [RangeError, NoMemoryError].each do |error|
        it "propagates #{error} without direct retry", :aggregate_failures do
          allow(model).to receive(:fft_time).and_raise(error, 'injected failure')
          expect { execution.automatic }.to raise_error(error, 'injected failure')
          expect(execution).not_to have_received(:basic)
        end
      end
    end

    it 'retains original estimates on unknown platforms', :aggregate_failures do
      stub_const('Recalibration::Profiles::CURRENT', nil)
      expect(execution.basic_time).to eq Convolver.predict_convolve_basic_time(signal, kernel)
      expect(execution.fft_time).to eq Convolver.predict_convolve_fft_time(signal, kernel)
    end

    it 'retains original estimates above rank three', :aggregate_failures do
      inputs = [Numo::DFloat.ones(5, 6, 7, 8), Numo::DFloat.ones(3, 3, 3, 3)]
      instance = described_class.new(:convolution, *inputs, **options)
      expect(instance.basic_time).to eq Convolver.predict_convolve_basic_time(*inputs)
      expect(instance.fft_time).to eq Convolver.predict_convolve_fft_time(*inputs)
    end

    def random_bounds(random, profile)
      inputs = random_inputs(random)
      plan = Convolver.const_get(:OperationPlan).new(*inputs, operation: :correlation, **random_options(random))
      model = CostModel.new(:correlation, *inputs, plan, profile)
      [model.lower_bound(Float::INFINITY), model.fft_time]
    end

    def options
      { mode: :valid, boundary: :constant, fill_value: Convolver.const_get(:UNSPECIFIED_FILL), origin: 0, dtype: nil }
    end

    def random_options(random)
      mode = %i[valid same full].sample(random:)
      boundary = mode == :same ? %i[constant nearest reflect mirror wrap].sample(random:) : :constant
      options.merge(mode:, boundary:)
    end

    def random_inputs(random)
      shape = Array.new(random.rand(1..3)) { random.rand(2..30) }
      [Numo::SFloat.new(*shape), Numo::DFloat.new(*shape.map { |size| random.rand(1..size) })]
    end
  end
end
