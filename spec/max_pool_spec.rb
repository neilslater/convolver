require 'helpers'

describe Convolver do
  describe "#max_pool" do
    describe "on a 1D NArray" do

      it "should return a copy of original array when tile and pool are 1" do
        input = NArray[ 1.0 ]
        output = Convolver.max_pool( input, 1, 1 )
        output.should be_narray_like NArray[ 1.0 ]

        input = NArray[ 1.0, 1.1, 1.2, -0.5 ]
        output = Convolver.max_pool( input, 1, 1 )
        output.should be_narray_like NArray[ 1.0, 1.1, 1.2, -0.5 ]
      end

      it "should reduce size of array by tile size, using maximum in each pool" do
        input = NArray[ 1.0, 1.1, 1.2, -0.5 ]
        output = Convolver.max_pool( input, 2, 2 )
        output.should be_narray_like NArray[ 1.1, 1.2]

        input = NArray[ 1.0, -1.1, -1.2, 0.5, 7.3, 2.0 ]
        output = Convolver.max_pool( input, 2, 2 )
        output.should be_narray_like NArray[ 1.0, 0.5, 7.3]
      end

      it "should allow pool size larger than tile size" do
        input = NArray[ 1.0, 1.1, 1.2, -0.5 ]
        output = Convolver.max_pool( input, 2, 3 )
        output.should be_narray_like NArray[ 1.2, 1.2]

        input = NArray[ 1.0, -1.1, -1.2, 0.5, 7.3, 2.0 ]
        output = Convolver.max_pool( input, 2, 4 )
        output.should be_narray_like NArray[ 1.0, 7.3, 7.3]
      end

      it "should allow tile size that is not exact fit to input" do
        input = NArray[ 1.0, 1.1, 1.2, -0.5 ]
        output = Convolver.max_pool( input, 3, 3 )
        output.should be_narray_like NArray[ 1.2, -0.5]

        input = NArray[ 1.0, -1.1, -1.2, 0.5, 7.3, 2.0 ]
        output = Convolver.max_pool( input, 4, 4 )
        output.should be_narray_like NArray[ 1.0, 7.3]
      end
    end # 1D array

    describe "on a 2D NArray" do

      it "should return a copy of original array when tile and pool are 1" do
        input = NArray[ [1.0] ]
        output = Convolver.max_pool( input, 1, 1 )
        output.should be_narray_like NArray[ [1.0] ]

        input = NArray[ [1.0, 1.1], [1.2, -0.5] ]
        output = Convolver.max_pool( input, 1, 1 )
        output.should be_narray_like NArray[ [1.0, 1.1], [1.2, -0.5] ]
      end

      it "should reduce size of array by tile size, using maximum in each pool" do
        input = NArray[ [1.0, 1.1, 1.2, -0.5], [1.3, -1.1, 1.0, -0.75],
            [-1.0, -1.1, -1.2, 0.5], [-1.3, 1.1, -1.0, 0.75] ]
        output = Convolver.max_pool( input, 2, 2 )
        output.should be_narray_like NArray[ [1.3, 1.2], [1.1, 0.75] ]

        input = NArray[ [  1.0, -1.1, -1.2,  0.5,  1.3,  2.0 ],
                        [  1.0, -1.1,  1.2,  0.5, -7.3,  2.1 ],
                        [  1.0, -1.1, -1.2,  2.5,  4.5, -2.0 ],
                        [ -1.0, -1.1, -1.2,  0.5,  1.1,  7.0 ],
                        [ -9.0, -0.1, -1.2,  1.5,  7.3,  0.0 ],
                        [ -1.0, -1.1,  0.0,  0.5,  1.2,  1.0 ] ]
        output = Convolver.max_pool( input, 3, 3 )
        output.should be_narray_like NArray[ [ 1.2, 4.5 ], [ 0.0, 7.3 ] ]
      end

      it "should allow pool size larger than tile size" do
        input = NArray[ [  1.0, -1.1, -1.2,  0.5,  1.3,  2.0 ],
                        [  1.0, -1.1,  1.2,  0.5, -7.3,  2.1 ],
                        [  1.0, -1.1, -1.2,  2.5,  4.5, -2.0 ],
                        [ -1.0, -1.1, -1.2,  0.5,  1.1,  7.0 ],
                        [ -9.0, -0.1, -1.2,  1.5,  7.3,  0.0 ],
                        [ -1.0, -1.1,  0.0,  0.5,  1.2,  1.0 ] ]
        output = Convolver.max_pool( input, 2, 3 )
        output.should be_narray_like NArray[ [ 1.2, 4.5, 4.5 ], [ 1.0, 7.3, 7.3 ], [ 0.0, 7.3, 7.3 ] ]
      end

      it "should allow tile size that is not exact fit to input" do
        input = NArray[ [ 1.0, 1.1, 1.2, -0.5 ] ]
        output = Convolver.max_pool( input, 3, 3 )
        output.should be_narray_like NArray[ [ 1.2, -0.5] ]

        input = NArray[ [ 1.0, -1.1, -1.2 ], [ 2.0, -2.1, -2.2 ], [ 1.0, -1.1, 7.3 ] ]
        output = Convolver.max_pool( input, 2, 2 )
        output.should be_narray_like NArray[ [ 2.0, -1.2 ], [ 1.0, 7.3 ] ]
      end
    end # 2D array

  end
end
