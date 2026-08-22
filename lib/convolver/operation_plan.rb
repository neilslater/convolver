# frozen_string_literal: true

require 'forwardable'
require 'convolver/operation_options'
require 'convolver/operation_shapes'
require 'convolver/signal_extension'

# Internal planning and extension support for Convolver's public operations.
module Convolver
  # Distinguishes an omitted fill_value keyword from an explicitly supplied
  # value. This lets non-constant boundaries reject even an explicit zero.
  UNSPECIFIED_FILL = Object.new.freeze

  # Validated dimensions and boundary-extension details for one operation.
  class OperationPlan
    extend Forwardable

    def_delegators :options, :operation, :mode, :boundary, :fill_value, :origins, :anchors
    def_delegators :shapes, :padding_before, :padding_after, :result_shape,
                   :extended_shape, :result_size, :extended_size, :linear_fft_shape,
                   :linear_fft_size, :linear_spectrum_size

    def initialize(signal, kernel, operation:, mode:, boundary:, fill_value:, origin:)
      @options = OperationOptions.new(signal, kernel, operation:, mode:, boundary:, fill_value:, origin:)
      @shapes = OperationShapes.new(
        signal.shape, kernel.shape, operation:, mode:, anchors: options.anchors
      )
    end

    def valid?
      mode == :valid
    end

    def wrap?
      mode == :same && boundary == :wrap
    end

    def extend_signal(signal)
      return signal if valid?

      SignalExtension.new(shapes, boundary:, fill_value:).call(signal)
    end

    private

    attr_reader :options, :shapes
  end

  private_constant :OperationPlan, :UNSPECIFIED_FILL
end
