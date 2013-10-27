require 'helpers'

describe Convolver do
  describe "#nn_run_layer" do
    it "should calculate basic layer rules" do
      inputs = NArray[ 1.0 ]
      weights = NArray[ [ 1.0 ] ]
      thresholds = NArray[ 0.0 ]
      outputs = Convolver.nn_run_layer( inputs, weights, thresholds );
      outputs.should be_narray_like NArray[ 1.0 ]

      inputs = NArray[ 0.5, -0.5 ]
      weights = NArray[ [ 1.0, 2.0 ], [ 2.0, 1.0 ] ]
      thresholds = NArray[ 0.0, 0.0 ]
      outputs = Convolver.nn_run_layer( inputs, weights, thresholds );
      outputs.should be_narray_like NArray[ 0.0, 0.5 ]

      inputs = NArray[ 0.3, -0.4, 0.8, -0.7 ]
      weights = NArray[ [ 1.0, 0.25, 0.5, -0.5 ], [ -1.0, -0.25, -0.5, 0.5 ] ]
      thresholds = NArray[ 0.0, 0.0 ]
      outputs = Convolver.nn_run_layer( inputs, weights, thresholds );
      outputs.should be_narray_like NArray[ 0.95, 0.0 ]
    end
  end
end
