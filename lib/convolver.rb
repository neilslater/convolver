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
    combined_shape, shift_by, ranges = fft_offsets( signal.shape, kernel.shape )

    mod_a = NArray.sfloat(*combined_shape)
    mod_a[*shift_by] = signal

    mod_b = NArray.sfloat(*combined_shape)

    Convolver.fit_kernel_backwards( mod_b, kernel )

    afreqs = FFTW3.fft(mod_a)
    bfreqs = FFTW3.fft(mod_b)
    cfreqs = afreqs * bfreqs

    (FFTW3.ifft( cfreqs ).real * (1.0/mod_a.size))[*ranges]
  end

  # A rough estimate of time that #convolve_fftw3 will take, based on complexity
  # of its operations, and some rough benchmarking. A value of 1.0 corresponds to results
  # varying between 1 and 12 milliseconds on the test computer.
  # @param [NArray] signal must be same size or larger than kernel in each dimension
  # @param [NArray] kernel must be same size or smaller than signal in each dimension
  # @return [Float] rough estimate of time for convolution compared to baseline
  def self.predict_convolve_fft_time signal, kernel
    16 * 4.55e-08 * combined_shape(signal.shape,kernel.shape).inject(1) { |t,x| t * x * Math.log(x) }
  end

  # A rough estimate of time that #convolve will take, based on complexity
  # of its operations, and some rough benchmarking. A value of 1.0 corresponds to results
  # varying bewteen 2 and 8 milliseconds on the test computer.
  # @param [NArray] signal must be same size or larger than kernel in each dimension
  # @param [NArray] kernel must be same size or smaller than signal in each dimension
  # @return [Float] rough estimate of time for convolution compared to baseline
  def self.predict_convolve_basic_time signal, kernel
    outputs = shape_to_size( result_shape( signal.shape, kernel.shape ) )
    4.54e-12 * (outputs * shape_to_size( signal.shape ) * shape_to_size( kernel.shape ))
  end

  private

  def self.shape_to_size shape
    shape.inject(1) { |t,x| t * x }
  end

  def self.combined_shape signal_shape, kernel_shape
    combined_shape = [  ]
    signal_shape.each_with_index do |signal_size, i|
      kernel_size = kernel_shape[i]
      combined_shape[i] = signal_size + kernel_size - 1
    end
    combined_shape
  end

  def self.result_shape signal_shape, kernel_shape
    result_shape = [  ]
    signal_shape.each_with_index do |signal_size, i|
      kernel_size = kernel_shape[i]
      result_shape[i] = signal_size - kernel_size + 1
    end
    result_shape
  end

  def self.fft_offsets signal_shape, kernel_shape
    combined_shape = []
    shift_by = []
    ranges = []
    signal_shape.each_with_index do |signal_size, i|
      kernel_size = kernel_shape[i]

      combined_shape[i] = signal_size + kernel_size - 1
      output_size = signal_size - kernel_size + 1
      output_offset = kernel_size - 1
      shift_by[i] = kernel_size / 2
      ranges[i] = (output_offset...(output_offset + output_size))
    end
    [ combined_shape, shift_by, ranges ]
  end
end
