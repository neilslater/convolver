# frozen_string_literal: true

require 'helpers'

module Convolver
  describe CostProfiles do
    { 'arm64-darwin25' => described_class::MAC,
      'x86_64-linux' => described_class::X86,
      'aarch64-linux' => described_class::ARM }.each do |platform, profile|
      it "selects the measured profile for #{platform}" do
        expect(described_class.for_platform(platform)).to equal(profile)
      end
    end

    %w[x64-mingw-ucrt x86_64-darwin25 x86_64-freebsd aarch64-unknown].each do |platform|
      it "retains the fallback for #{platform}" do
        expect(described_class.for_platform(platform)).to be_nil
      end
    end

    [described_class::MAC, described_class::X86, described_class::ARM, nil].each_with_index do |profile, index|
      context "with profile #{index}" do
        before { stub_const('Convolver::CostProfiles::CURRENT', profile) }

        [0, 1, 2, 3, 4, 16].each do |rank|
          it "limits calibration to ranks one through three for rank #{rank}" do
            expect(described_class.for_rank(rank)).to equal((1..3).cover?(rank) ? profile : nil)
          end
        end

        %i[convolve correlate].each do |method|
          it "resolves the profile once during #{method}" do
            allow(described_class).to receive(:for_rank).and_call_original
            Convolver.public_send(method, Numo::SFloat.ones(999), Numo::SFloat.ones(500))
            expect(described_class).to have_received(:for_rank).with(1).once
          end
        end
      end
    end

    [described_class::MAC, described_class::X86, described_class::ARM].each_with_index do |profile, index|
      it "freezes profile #{index} and its nested coefficient tables" do
        expect(nested_tables(profile)).to all(be_frozen)
      end
    end

    def nested_tables(value)
      return [] unless value.is_a?(Hash) || value.is_a?(Array)

      children = value.is_a?(Hash) ? value.values : value
      [value] + children.flat_map { |child| nested_tables(child) }
    end
  end
end
