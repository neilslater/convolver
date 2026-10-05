# frozen_string_literal: true

require 'helpers'
require 'dtype_fixtures'

module Convolver
  describe Convolver do
    let(:signal) { Numo::DFloat.ones(1024) }
    let(:kernel) { Numo::SFloat.ones(8) }

    OperationReference::ESTIMATOR_METHODS.values.flatten.each do |method_name|
      it "estimates every accepted input pair without casting in #{method_name}", :aggregate_failures do
        inputs = DtypeFixtures::INPUTS.keys.map { |type| type[1, 2] }
        [Numo::SFloat, Numo::DFloat].each { |type| allow(type).to receive(:cast).and_call_original }
        inputs.product(inputs).each { |pair| expect(described_class.public_send(method_name, *pair)).to be_positive }
        expect(Numo::SFloat).not_to have_received(:cast)
        expect(Numo::DFloat).not_to have_received(:cast)
      end

      it "charges for the explicit input conversion in #{method_name}" do
        matched = described_class.public_send(method_name, kernel, kernel)
        converted = described_class.public_send(method_name, Numo::Int8.cast(kernel), kernel)
        expect(converted).to be > matched
      end
    end

    %i[convolve correlate].each do |method_name|
      it "resolves dtype once across #{method_name} automatic planning and execution" do
        allow(OperationDtype).to receive(:resolve).and_call_original
        described_class.public_send(method_name, signal, kernel)
        expect(OperationDtype).to have_received(:resolve).once
      end
    end
  end
end
