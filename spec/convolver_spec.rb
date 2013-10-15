require 'helpers'

describe Convolver do
  describe "#convolve" do
    it "should work like example in documentation" do
      a = NArray[ 0.3, 0.4, 0.5 ]
      b = NArray[ 1.3, -0.5 ]
      c = Convolver.convolve( a, b )
      c.should be_narray_like NArray[ 0.19, 0.27 ]
    end

    it "should handle a 2D convolution" do
      a = NArray[ [ 0.3, 0.4, 0.5 ], [ 0.6, 0.8, 0.2 ], [ 0.9, 1.0, 0.1 ] ]
      b = NArray[ [ 1.2, -0.5 ], [ 0.5, -1.3 ] ]
      c = Convolver.convolve( a, b )
      c.should be_narray_like NArray[ [ -0.58, 0.37 ], [ -0.53, 1.23 ] ]
    end

  end
end
