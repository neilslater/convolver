require 'narray'
require "convolver/convolver"
require "convolver/version"
require 'fftw3'

module Convolver
  # Uses FFTW3 library to calculates convolution of an array of floats representing a signal,
  # with a second array representing a kernel. The two parameters must have the same rank.
  # The output has same rank, its size in each dimension d is given by
  #  signal.shape[d] - kernel.shape[d] + 1
  # @param [NArray] signal must be same size or larger than kernel in each dimension
  # @param [NArray] kernel must be same size or smaller than signal in each dimension
  # @return [NArray] result of convolving signal with kernel
  def self.convolve_fftw3 signal, kernel
    combined_size = signal.size + kernel.size - 1
    output_size = signal.size - kernel.size + 1
    output_offset = kernel.size - 1

    mod_a = NArray.float(combined_size)
    left_pad_signal = ( combined_size - signal.size + 1 )/2
    mod_a[left_pad_signal] = signal

    mod_b = NArray.float(combined_size)
    left_shift_kernel = kernel.size / 2
    b_rev = kernel.reverse
    mod_b[0] = b_rev[(left_shift_kernel...kernel.size)]
    mod_b[-left_shift_kernel] = b_rev[0...left_shift_kernel] if left_shift_kernel > 0

    afft = FFTW3.fft(mod_a)
    bfft = FFTW3.fft(mod_b)
    cfft = afft * bfft

    (FFTW3.ifft( cfft )/combined_size).real[output_offset...(output_offset + output_size)]
  end

  private

  def self.fft_offsets signal_size, kernel_size
    combined_size = signal_size + kernel_size - 1
    output_size = signal_size - kernel_size + 1
    output_offset = kernel_size - 1
    left_pad_signal = ( combined_size - signal_size + 1 )/2
    left_shift_kernel = kernel.size / 2
  end
end
