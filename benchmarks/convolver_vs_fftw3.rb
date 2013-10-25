require 'convolver'
require 'benchmark'

class Convolver2DBenchmark
  attr_reader :image, :kernel

  def initialize
    # These show Convolver.convolve as 3x faster than FFTW3
    @image = NArray.sfloat(256 * 256).random
    @kernel = NArray.sfloat(16 * 16).random

    # These are roughly even (10% advantage to FFTW3)
    #  @image = NArray.sfloat(256 * 256).random
    #  @kernel = NArray.sfloat(32 * 32).random

    # These show FFTW3 as 4x faster than Convolver.convolve
    #  @image = NArray.sfloat(256 * 256).random
    #  @kernel = NArray.sfloat(64 * 64).random

    # These show Convolver.convolve as 200x faster than FFTW3
    # @image = NArray.sfloat(50 * 64 * 64).random
    # @kernel = NArray.sfloat(50 * 64 * 64).random

    # These show FFTW3 as 2x faster than Convolver.convolve
    # @image = NArray.sfloat(128 * 128).random
    # @kernel = NArray.sfloat(64 * 64).random

    # These show FFTW3 and Convolver.convolve roughly equal
    # @image = NArray.sfloat(80 * 80).random
    # @kernel = NArray.sfloat(64 * 64).random

    # These show FFTW3 as 2x faster than Convolver.convolve
    # @image = NArray.sfloat(2 * 80 * 80).random
    # @kernel = NArray.sfloat(2 * 64 * 64).random

    # These are roughly even - increasing size of image favours FFTW3
    #@image = NArray.sfloat(2000 + 80 * 80).random
    #@kernel = NArray.sfloat(80 * 80).random
  end
end

Benchmark.bm do |x|
  source = Convolver2DBenchmark.new
  x.report('convolver') { 100.times { Convolver.convolve( source.image, source.kernel ) } }
  x.report('fftw3') { 100.times { Convolver.convolve_fftw3( source.image, source.kernel ) } }
end
