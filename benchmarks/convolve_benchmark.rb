require 'convolver'
require 'narray'
require 'benchmark'

class Convolver2DBenchmark
  attr_reader :image, :kernel

  def initialize
    @image = NArray.float(640, 480).random
    @kernel = NArray.float(8, 8).random
  end
end

Benchmark.bm do |x|
  source = Convolver2DBenchmark.new
  x.report('kilo') { 1000.times { Convolver.convolve( source.image, source.kernel ) } }
end

