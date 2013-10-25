require 'narray'
require "convolver/convolver"
require "convolver/version"
require 'fftw3'

module Convolver
  # Uses FFTW3 library to calculate convolution of an array of floats representing a signal,
  # with a second array representing a kernel. The two parameters must have the same rank.
  # The output has same rank, its size in each dimension d is given by
  #  signal.shape[d] - kernel.shape[d] + 1
  # @param [NArray] signal must be same size or larger than kernel in each dimension
  # @param [NArray] kernel must be same size or smaller than signal in each dimension
  # @return [NArray] result of convolving signal with kernel
  def self.convolve_fftw3 signal, kernel
    combined_shape, output_shape, output_offset, shift_by, ranges = fft_offsets( signal.shape, kernel.shape )

    mod_a = NArray.sfloat(*combined_shape)
    mod_a[*shift_by] = signal

    mod_b = NArray.sfloat(*combined_shape)

    Convolver.fit_kernel_backwards( mod_b, kernel )

    afft = FFTW3.fft(mod_a)
    bfft = FFTW3.fft(mod_b)
    cfft = afft * bfft

    (FFTW3.ifft( cfft )/mod_a.size).real[*ranges]
  end

  private

  def self.fft_offsets signal_shape, kernel_shape
    combined_shape = []
    output_shape = []
    output_offset = []
    shift_by = []
    ranges = []
    signal_shape.each_with_index do |signal_size, i|
      kernel_size = kernel_shape[i]

      combined_shape[i] = signal_size + kernel_size - 1
      output_shape[i] = signal_size - kernel_size + 1
      output_offset[i] = kernel_size - 1
      shift_by[i] = kernel_size / 2
      ranges[i] = (output_offset[i]...(output_offset[i] + output_shape[i]))
    end
    [ combined_shape, output_shape, output_offset, shift_by, ranges ]
  end
end
