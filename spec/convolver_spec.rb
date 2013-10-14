require 'convolver'

describe Convolver do
  describe "#ext_test" do
    it "should return 80193" do
      Convolver.ext_test.should == 80193
    end
  end
end
