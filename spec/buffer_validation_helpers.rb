# frozen_string_literal: true

module Convolver
  # Small injected architectural limits exercise public calls without large arrays.
  module BufferValidationHelpers
    def inject_limits(byte_max: BufferLimits::BYTE_MAX, int_max: BufferLimits::INT_MAX)
      stub_const('Convolver::BufferLimits::NATIVE', BufferLimits.new(byte_max:, int_max:))
    end

    def allocation_receivers
      { SignalExtension => [:new], Numo::SFloat => [:cast],
        Numo::DFloat => %i[cast zeros], Numo::Pocketfft => %i[rfftn fftn] }
    end

    def watch_allocations
      allocation_receivers.each do |receiver, methods|
        methods.each { |method| allow(receiver).to receive(method).and_call_original }
      end
    end

    def expect_no_allocations
      allocation_receivers.each do |receiver, methods|
        methods.each { |method| expect(receiver).not_to have_received(method) }
      end
    end
  end
end
