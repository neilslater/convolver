require 'convolver'
require 'narray'
require 'fftw3'
require 'benchmark'

# In Ruby for now, which is slower, but at least gets us ballpark figures
module FFTW3Convolver
  def self.convolve orig_a, orig_b
    combined_size = orig_a.size + orig_b.size - 1
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

    (FFTW3.ifft( cfft )/combined_size).real[left_pad_a...(left_pad_a+ orig_a.size - orig_b.size + 1)]
  end
end

class Convolver2DBenchmark
  attr_reader :image, :kernel

  def initialize
    @image = NArray.float(256 * 256).random
    @kernel = NArray.float(4 * 256).random
  end
end

Benchmark.bm do |x|
  source = Convolver2DBenchmark.new
  x.report('convolver') { 100.times { Convolver.convolve( source.image, source.kernel ) } }
  x.report('fftw3') { 100.times { FFTW3Convolver.convolve( source.image, source.kernel ) } }
end
