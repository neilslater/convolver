require 'convolver'

describe Convolver do
  describe "#convolve" do
    it "should work like example in documentation" do
      a = NArray[0.3,0.4,0.5]
      b = NArray[1.3, -0.5]
      c = Convolver.convolve( a, b )
      c.should be_a NArray
      p c
      c[0].should be_within(1e-10).of 0.19
      c[1].should be_within(1e-10).of 0.27
    end
  end
end
