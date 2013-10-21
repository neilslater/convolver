require 'convolver'
require 'narray'
require 'benchmark'

class ConvolverNNLayerBenchmark
  attr_reader :input, :weights, :thresholds

  def initialize
    @input = NArray.float(1024).random
    @weights = NArray.float(1024,256).random
    @thresholds = NArray.float(256).random
  end
end

Benchmark.bm do |x|
  source = ConvolverNNLayerBenchmark.new
  x.report('kilo') { 1000.times { Convolver.nn_run_layer( source.input, source.weights, source.thresholds ) } }
end
