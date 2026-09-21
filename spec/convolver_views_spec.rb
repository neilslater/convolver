# frozen_string_literal: true

require 'helpers'
require 'native_view_fixtures'

describe Convolver do
  shared_examples 'logical view values' do |operation, mode|
    let(:sources) { NativeViewFixtures.backing_arrays }
    let(:signal) { views.first }
    let(:kernel) { views.last }
    let(:expected) { OperationReference.calculate(operation, signal, kernel, mode:) }
    let(:originals) { sources.map(&:to_a) }

    before do
      originals
      signal.freeze
      kernel.freeze
      sources.each(&:freeze)
    end

    OperationReference::CALCULATION_METHODS.fetch(operation).each do |method_name|
      it "calculates logical values through .#{method_name} in #{mode} mode" do
        result = described_class.public_send(method_name, signal, kernel, mode:)

        expect(result).to be_narray_like(expected, 1e-9)
      end
    end

    it "preserves backing arrays and frozen views in #{mode} mode" do
      methods = OperationReference::CALCULATION_METHODS.fetch(operation)
      methods.each { |method_name| described_class.public_send(method_name, signal, kernel, mode:) }

      expect(sources.map(&:to_a)).to eq originals
    end
  end

  OperationReference::OPERATIONS.product(%i[valid same full]).each do |operation, mode|
    layouts = %i[signal_offset kernel_offset both_offsets zero_offset reversed indexed]
    layouts.product([3, 4, 5, 8]).each do |layout, length|
      context "with #{layout}, kernel length #{length}, #{operation}, and #{mode} mode" do
        let(:views) { NativeViewFixtures.views(layout, *sources, length) }

        it_behaves_like 'logical view values', operation, mode
      end
    end

    %i[multidimensional transposed scalar].each do |layout|
      context "with #{layout} views, #{operation}, and #{mode} mode" do
        let(:views) { NativeViewFixtures.shaped_views(layout, *sources) }

        it_behaves_like 'logical view values', operation, mode
      end
    end

    context "with double precision offset views, #{operation}, and #{mode} mode" do
      let(:sources) { NativeViewFixtures.backing_arrays(Numo::DFloat) }
      let(:views) { NativeViewFixtures.views(:both_offsets, *sources, 5) }

      it_behaves_like 'logical view values', operation, mode
    end
  end

  context 'with the original contiguous slice regression' do
    let(:signal) { NArray[100, 2, 3, 4][1..3] }
    let(:kernel) { NArray[100, 1, 2][1..2] }

    it 'uses contiguous views' do
      expect([signal, kernel]).to all(be_contiguous)
    end

    it 'convolves the logical values' do
      expect(described_class.convolve_basic(signal, kernel)).to be_narray_like(NArray[7, 10], 1e-9)
    end

    it 'correlates the logical values' do
      expect(described_class.correlate_basic(signal, kernel)).to be_narray_like(NArray[8, 11], 1e-9)
    end
  end
end
