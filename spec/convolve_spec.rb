require 'helpers'

describe Convolver do
  describe "#convolve" do

    it "should work like the example in the README" do
      a = NArray[ 0.3, 0.4, 0.5 ]
      b = NArray[ 1.3, -0.5 ]
      c = Convolver.convolve( a, b )
      c.should be_narray_like NArray[ 0.19, 0.27 ]
    end

  end
end
