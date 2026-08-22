# frozen_string_literal: true

module Convolver
  # Calculates same-sized periodic operations through PocketFFT transforms.
  class CircularFftOperation
    def initialize(operation, signal, kernel, plan)
      @operation = operation
      @signal = Numo::DFloat.cast(signal)
      @kernel = fold_kernel(Numo::DFloat.cast(kernel), signal.shape, plan.anchors)
    end

    def call
      real_axis = self.class.real_axis(signal.shape)
      return complex_product unless real_axis

      real_product(real_axis)
    end

    def self.real_axis(shape)
      return nil if shape.empty?

      final_axis = shape.length - 1
      return final_axis if shape[final_axis].even?

      shape.each_index.select { |axis| shape[axis].even? }.max_by { |axis| shape[axis] }
    end

    private

    attr_reader :operation, :signal, :kernel

    def real_product(real_axis)
      final_axis = signal.ndim - 1
      prepared_signal = prepare_real_operand(signal, real_axis, final_axis)
      prepared_kernel = prepare_real_operand(kernel, real_axis, final_axis)
      result = real_spectrum_product(prepared_signal, prepared_kernel)
      result = result.swapaxes(real_axis, final_axis) unless real_axis == final_axis
      Numo::SFloat.cast(result)
    end

    def prepare_real_operand(value, real_axis, final_axis)
      real_axis == final_axis ? value : value.swapaxes(real_axis, final_axis)
    end

    def real_spectrum_product(signal_value, kernel_value)
      signal_spectrum = Numo::Pocketfft.rfftn(signal_value)
      kernel_spectrum = Numo::Pocketfft.rfftn(kernel_value)
      kernel_spectrum.inplace.conj if operation == :correlation
      Numo::Pocketfft.irfftn(signal_spectrum.inplace * kernel_spectrum)
    end

    def complex_product
      signal_spectrum = Numo::Pocketfft.fftn(signal)
      kernel_spectrum = Numo::Pocketfft.fftn(kernel)
      kernel_spectrum.inplace.conj if operation == :correlation
      result = Numo::Pocketfft.ifftn(signal_spectrum.inplace * kernel_spectrum).real
      Numo::SFloat.cast(result)
    end

    def fold_kernel(source, signal_shape, anchors)
      folded = Numo::DFloat.zeros(*signal_shape)
      source.flatten.each_with_index do |value, flat_index|
        target = folded_target(flat_index, source.shape, signal_shape, anchors)
        folded[*target] = folded[*target] + value
      end
      folded
    end

    def folded_target(flat_index, kernel_shape, signal_shape, anchors)
      remainder = flat_index
      Array.new(kernel_shape.length).tap do |target|
        (kernel_shape.length - 1).downto(0) do |axis|
          coordinate = remainder % kernel_shape[axis]
          remainder /= kernel_shape[axis]
          target[axis] = (coordinate - anchors[axis]) % signal_shape[axis]
        end
      end
    end
  end

  private_constant :CircularFftOperation
end
