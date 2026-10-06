# frozen_string_literal: true

require 'helpers'

module Convolver
  describe FftCost do
    [CostProfiles::MAC, CostProfiles::X86, CostProfiles::ARM, nil].each_with_index do |profile, index|
      context "with profile #{index}" do
        before do
          stub_const('Convolver::CostProfiles::CURRENT', profile)
          allow(FftBufferPlan).to receive(:new).and_call_original
        end

        [0, 1, 2, 3, 4].product(%i[constant nearest reflect mirror wrap]).each do |rank, boundary|
          it "bounds rank #{rank} #{boundary} estimation without planning", :aggregate_failures do
            inputs = [Numo::DFloat.new(*([4] * rank)), Numo::SFloat.new(*([3] * rank))]
            plan = build_plan(*inputs, boundary:)
            lower = described_class.lower_bound(plan, inputs.last.shape, threshold: Float::INFINITY)
            expect(FftBufferPlan).not_to have_received(:new)
            expect(lower).to be <= FftEstimator.new(:convolution, *inputs, plan).call
          end
        end

        it 'uses the setup floor before deriving transform dimensions' do
          plan = instance_double(OperationPlan, cost_profile: CostProfiles.for_rank(1))
          expected = described_class.fixed(1, profile: plan.cost_profile)
          expect(described_class.lower_bound(plan, [3], threshold: 0)).to eq expected
        end

        it 'keeps cost arithmetic representable when mandatory FFT dimensions overflow' do
          plan = build_plan(Numo::SFloat.new(4), Numo::SFloat.new(3))
          allow(plan).to receive(:extended_shape).and_return([BufferLimits::SIZE_MAX])
          expect(described_class.lower_bound(plan, [3], threshold: Float::INFINITY)).to be_finite
        end

        it 'bounds seeded valid and full transforms' do
          random = Random.new(53_617)
          100.times do
            lower, full = random_bounds(random)
            expect(lower).to be <= full
          end
        end

        def random_bounds(random)
          inputs = random_inputs(random)
          plan = build_plan(*inputs, mode: %i[valid full].sample(random:))
          lower = described_class.lower_bound(plan, inputs.last.shape, threshold: Float::INFINITY)
          [lower, FftEstimator.new(:correlation, *inputs, plan).call]
        end

        def build_plan(signal, kernel, mode: :same, boundary: :constant)
          OperationPlan.new(signal, kernel, operation: :convolution, mode:, boundary:,
                                            fill_value: UNSPECIFIED_FILL, origin: 0)
        end

        def random_inputs(random)
          shape = Array.new(random.rand(1..3)) { random.rand(2..50) }
          kernel_shape = shape.map { |size| random.rand(1..size) }
          [Numo::SFloat.new(*shape), Numo::DFloat.new(*kernel_shape)]
        end
      end
    end
  end
end
