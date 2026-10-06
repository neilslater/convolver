# frozen_string_literal: true

require 'forwardable'
require 'convolver/cost_profiles'
require 'convolver/operation_dtype'
require 'convolver/operation_options'
require 'convolver/operation_shapes'
require 'convolver/signal_extension'
require 'convolver/fft_buffer_plan'

# Internal planning and extension support for Convolver's public operations.
module Convolver
  # Distinguishes an omitted fill_value keyword from an explicitly supplied
  # value. This lets non-constant boundaries reject even an explicit zero.
  # @private
  UNSPECIFIED_FILL = Object.new.freeze

  # Validated dimensions and boundary-extension details for one operation.
  # @private
  class OperationPlan
    extend Forwardable

    def_delegators :options, :operation, :mode, :boundary, :fill_value, :origins, :anchors, :dtype
    def_delegators :shapes, :padding_before, :padding_after, :result_shape,
                   :extended_shape, :result_size, :extended_size, :linear_fft_shape,
                   :linear_fft_size, :linear_spectrum_size

    attr_reader :conversion_size, :cost_profile

    def initialize(signal, kernel, operation:, mode:, boundary:, fill_value:, origin:, dtype: nil)
      @options = OperationOptions.new(signal, kernel, operation:, mode:, boundary:, fill_value:, origin:, dtype:)
      @conversion_size = input_conversion_size(signal, kernel)
      @signal_shape = signal.shape.freeze
      @kernel_shape = kernel.shape.freeze
      @cost_profile = CostProfiles.for_rank(@signal_shape.length)
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

    def validate_basic!
      return if @basic_validated

      BufferPreparation.new(self, @signal_shape, @kernel_shape).basic!
      @basic_validated = true
    end

    def fft_buffers
      @fft_buffers ||= FftBufferPlan.new(self, @signal_shape, @kernel_shape)
    end

    private

    attr_reader :options, :shapes

    def input_conversion_size(signal, kernel)
      [signal, kernel].sum { |input| input.instance_of?(dtype) ? 0 : input.size }
    end
  end

  private_constant :OperationPlan, :UNSPECIFIED_FILL
end
