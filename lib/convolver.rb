# frozen_string_literal: true

require 'numo/narray/alt'
require 'numo/pocketfft'
require 'convolver/convolver'
require 'convolver/version'

# Cross-correlation operations for Numo::NArray values.
module Convolver
  # Maximum number of dimensions supported by the implementations.
  MAX_RANK = 16

  require 'convolver/operation_plan'

  class << self
    # Chooses the likely fastest cross-correlation implementation.
    #
    # @param signal [Numo::NArray] input values
    # @param kernel [Numo::NArray] correlation kernel
    # @param mode [:valid, :same, :full] returned output extent
    # @param boundary [:constant, :nearest, :reflect, :mirror, :wrap] signal extension
    # @param fill_value [Numeric] constant extension value
    # @param origin [Integer, Array<Integer>] kernel origin shift
    # @return [Numo::SFloat] cross-correlation result
    # @raise [ArgumentError] if inputs or options are incompatible
    def convolve(signal, kernel, mode: :valid, boundary: :constant,
                 fill_value: UNSPECIFIED_FILL, origin: 0)
      plan = operation_plan(signal, kernel, mode:, boundary:, fill_value:, origin:)
      options = operation_options(mode, boundary, fill_value, origin)

      return convolve_basic(signal, kernel, **options) if plan.extended_size < 1000 || kernel.size < 100

      basic_time_predicted = predict_convolve_basic_time(signal, kernel, **options)
      return convolve_basic(signal, kernel, **options) if basic_time_predicted < 0.1

      fft_time_predicted = predict_convolve_fft_time(signal, kernel, **options)
      return convolve_fft(signal, kernel, **options) if fft_time_predicted < 2 * basic_time_predicted

      convolve_basic(signal, kernel, **options)
    end

    # Uses the direct native valid primitive after applying the requested signal
    # extension in Ruby.
    #
    # @return [Numo::SFloat] cross-correlation result
    def convolve_basic(signal, kernel, mode: :valid, boundary: :constant,
                       fill_value: UNSPECIFIED_FILL, origin: 0)
      plan = operation_plan(signal, kernel, mode:, boundary:, fill_value:, origin:)
      convolve_basic_valid(plan.extend_signal(signal), kernel)
    end

    # Uses PocketFFT to calculate the requested cross-correlation.
    #
    # Periodic same-sized results use a circular transform. Other combinations
    # use the shared extension plan and PocketFFT's linear convolution.
    #
    # @return [Numo::SFloat] cross-correlation result
    def convolve_fft(signal, kernel, mode: :valid, boundary: :constant,
                     fill_value: UNSPECIFIED_FILL, origin: 0)
      plan = operation_plan(signal, kernel, mode:, boundary:, fill_value:, origin:)
      return convolve_fft_wrap(signal, kernel, plan) if plan.wrap?

      convolve_fft_valid(plan.extend_signal(signal), kernel)
    end

    # Compatibility alias for the former FFTW3-backed implementation.
    #
    # @deprecated Use {.convolve_fft}; Convolver no longer uses FFTW3.
    # @return [Numo::SFloat] cross-correlation result
    def convolve_fftw3(signal, kernel, mode: :valid, boundary: :constant,
                       fill_value: UNSPECIFIED_FILL, origin: 0)
      warn 'Convolver.convolve_fftw3 is deprecated; use .convolve_fft instead', uplevel: 1
      options = operation_options(mode, boundary, fill_value, origin)
      convolve_fft(signal, kernel, **options)
    end

    # Estimates the relative cost of {.convolve_fft} for the requested options.
    #
    # @return [Float] machine-specific relative cost estimate
    def predict_convolve_fft_time(signal, kernel, mode: :valid, boundary: :constant,
                                  fill_value: UNSPECIFIED_FILL, origin: 0)
      plan = operation_plan(signal, kernel, mode:, boundary:, fill_value:, origin:)
      transform_size = plan.wrap? ? signal.size : plan.linear_fft_size(kernel.shape)
      transform_cost = 16 * 4.55e-08 * transform_size * Math.log(transform_size)
      transform_cost + (4.55e-08 * fft_preparation_size(plan, signal, kernel))
    end

    # Estimates the relative cost of {.convolve_basic} for the requested options.
    #
    # @return [Float] machine-specific relative cost estimate
    def predict_convolve_basic_time(signal, kernel, mode: :valid, boundary: :constant,
                                    fill_value: UNSPECIFIED_FILL, origin: 0)
      plan = operation_plan(signal, kernel, mode:, boundary:, fill_value:, origin:)
      operations = plan.result_size * plan.extended_size * kernel.size
      operations += plan.extended_size unless plan.valid?
      4.54e-12 * operations
    end

    private

    private :convolve_basic_valid

    def operation_plan(signal, kernel, mode:, boundary:, fill_value:, origin:)
      OperationPlan.new(signal, kernel, mode:, boundary:, fill_value:, origin:)
    end

    def operation_options(mode, boundary, fill_value, origin)
      options = { mode:, boundary:, origin: }
      options[:fill_value] = fill_value unless fill_value.equal?(UNSPECIFIED_FILL)
      options
    end

    def convolve_fft_valid(signal, kernel)
      ranges = kernel.shape.zip(signal.shape).map do |kernel_size, signal_size|
        (kernel_size - 1)...signal_size
      end
      full_convolution = Numo::Pocketfft.fftconvolve(signal, kernel.reverse)

      Numo::SFloat.cast(full_convolution[*ranges])
    end

    def convolve_fft_wrap(signal, kernel, plan)
      signal_value = Numo::DFloat.cast(signal)
      folded_kernel = fold_kernel(Numo::DFloat.cast(kernel), signal.shape, plan.anchors)
      spectrum = Numo::Pocketfft.fftn(signal_value) * Numo::Pocketfft.fftn(folded_kernel).conj

      Numo::SFloat.cast(Numo::Pocketfft.ifftn(spectrum).real)
    end

    def fold_kernel(kernel, signal_shape, anchors)
      folded = Numo::DFloat.zeros(*signal_shape)
      kernel.flatten.each_with_index do |value, flat_index|
        target = folded_kernel_target(flat_index, kernel.shape, signal_shape, anchors)
        folded[*target] = folded[*target] + value
      end
      folded
    end

    def folded_kernel_target(flat_index, kernel_shape, signal_shape, anchors)
      remainder = flat_index
      Array.new(kernel_shape.length).tap do |target|
        (kernel_shape.length - 1).downto(0) do |axis|
          coordinate = remainder % kernel_shape[axis]
          remainder /= kernel_shape[axis]
          target[axis] = (coordinate - anchors[axis]) % signal_shape[axis]
        end
      end
    end

    def fft_preparation_size(plan, signal, kernel)
      return signal.size + kernel.size if plan.wrap?
      return 0 if plan.valid?

      plan.extended_size
    end
  end
end
