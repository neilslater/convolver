require 'convolver'
require 'narray'
require 'fftw3'
require 'benchmark'

# In Ruby for now, which is slower, but at least gets us ballpark figures (99% of the work is in the C)
module FFTW3Convolver
  def self.convolve orig_a, orig_b
    combined_size = orig_a.size + orig_b.size - 1
    output_size = orig_a.size - orig_b.size + 1
    output_offset = ( orig_b.size )/2

    left_pad_a = ( combined_size - orig_a.size + 1)/2
    mod_a = NArray.float(combined_size)
    mod_a[left_pad_a] = orig_a

    mod_b = NArray.float(combined_size)
    left_select_b = ( orig_b.size + 1 )/2
    right_select_b = orig_b.size - left_select_b
    mod_b[0] = orig_b[(0...left_select_b)].reverse
    mod_b[-right_select_b] = orig_b[-right_select_b..-1].reverse

    afft = FFTW3.fft(mod_a)
    bfft = FFTW3.fft(mod_b)
    cfft = afft * bfft

    (FFTW3.ifft( cfft )/combined_size).real[output_offset...(left_pad_a+ orig_a.size - orig_b.size + 1)]
  end
end

class Convolver2DBenchmark
  attr_reader :image, :kernel

  def initialize
    # These show Convolver.convolve as 3x faster than FFTW3
    #  @image = NArray.float(256 * 256).random
    #  @kernel = NArray.float(16 * 16).random

    # These are roughly even (10% advantage to FFTW3)
    #  @image = NArray.float(256 * 256).random
    #  @kernel = NArray.float(32 * 32).random

    # These show FFTW3 as 4x faster than Convolver.convolve
    #  @image = NArray.float(256 * 256).random
    #  @kernel = NArray.float(64 * 64).random

    # These show Convolver.convolve as 200x faster than FFTW3
    # @image = NArray.float(50 * 64 * 64).random
    # @kernel = NArray.float(50 * 64 * 64).random

    # These show FFTW3 as 2x faster than Convolver.convolve
    # @image = NArray.float(128 * 128).random
    # @kernel = NArray.float(64 * 64).random

    # These show FFTW3 and Convolver.convolve roughly equal
    # @image = NArray.float(80 * 80).random
    # @kernel = NArray.float(64 * 64).random

    # These show FFTW3 as 2x faster than Convolver.convolve
    # @image = NArray.float(2 * 80 * 80).random
    # @kernel = NArray.float(2 * 64 * 64).random

    # These are roughly even - increasing size of image favours FFTW3
    @image = NArray.float(2000 + 80 * 80).random
    @kernel = NArray.float(80 * 80).random
  end
end

Benchmark.bm do |x|
  source = Convolver2DBenchmark.new
  x.report('convolver') { 100.times { Convolver.convolve( source.image, source.kernel ) } }
  x.report('fftw3') { 100.times { FFTW3Convolver.convolve( source.image, source.kernel ) } }
end
