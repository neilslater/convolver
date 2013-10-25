require 'helpers'

describe Convolver do
  describe "#convolve_fftw3" do

    it "should work like the example in the README" do
      a = NArray[ 0.3, 0.4, 0.5 ]
      b = NArray[ 1.3, -0.5 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ 0.19, 0.27 ]
    end

    it "should convolve 1D arrays with a variety of signal and kernel lengths" do
      a = NArray[ 0.3 ]
      b = NArray[ -0.7 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ -0.21 ]

      a = NArray[ 0.3, 0.4, 0.5, 0.2 ]
      b = NArray[ -0.7 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ -0.21, -0.28, -0.35, -0.14 ]

      a = NArray[ 0.3, 0.4, 0.5, 0.2 ]
      b = NArray[ 1.1, -0.7 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ 0.05, 0.09, 0.41 ]

      a = NArray[ 0.3, 0.4, 0.5, 0.2 ]
      b = NArray[ 1.1, -0.7, -0.2 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ -0.05, 0.05 ]

      a = NArray[ 0.3, 0.4, 0.5, 0.2, 0.6 ]
      b = NArray[ 1.1, -0.7 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ 0.05, 0.09, 0.41, -0.2 ]

      a = NArray[ 0.3, 0.4, 0.5, 0.2, 0.6 ]
      b = NArray[ 1.1, -0.7, 2.1 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ 1.1, 0.51, 1.67 ]

      a = NArray[ 0.3, 0.4, 0.5, 0.2, 0.6 ]
      b = NArray[ 0.6, -0.5, -0.4, 0.7 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ -0.08, 0.33 ]
    end

    it "should calculate a 2D convolution" do
      a = NArray[ [ 0.3, 0.4, 0.5 ], [ 0.6, 0.8, 0.2 ], [ 0.9, 1.0, 0.1 ] ]
      b = NArray[ [ 1.2, -0.5 ], [ 0.5, -1.3 ] ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ [ -0.58, 0.37 ], [ -0.53, 1.23 ] ]
    end

    it "should calculate a 3D convolution" do
      # 5x4x3
      a = NArray[
        [ [ 1.0, 0.6, 1.1, 0.2, 0.9 ], [ 1.0, 0.7, 0.8, 1.0, 1.0 ], [ 0.2, 0.6, 0.1, 0.2, 0.5 ], [ 0.5, 0.9, 0.2, 0.1, 0.6 ] ],
        [ [ 0.4, 0.9, 0.4, 0.0, 0.6 ], [ 0.2, 1.1, 0.2, 0.4, 0.1 ], [ 0.4, 0.2, 0.5, 0.8, 0.7 ], [ 0.1, 0.9, 0.7, 0.1, 0.3 ] ],
        [ [ 0.8, 0.6, 1.0, 0.1, 0.4 ], [ 0.3, 0.8, 0.6, 0.7, 1.1 ], [ 0.9, 1.0, 0.3, 0.4, 0.6 ], [ 0.2, 0.5, 0.4, 0.7, 0.2 ] ]
      ]

      # 3x3x3
      b = NArray[
        [ [ -0.9, 1.2, 0.8  ], [ 0.9, 0.1, -0.5 ], [ 1.1, 0.1, -1.1 ] ],
        [ [ -0.2, -1.0, 1.4 ], [ -1.4, 0.0, 1.3 ], [ 0.3, 1.0, -0.5 ] ],
        [ [ 0.6, 0.0, 0.7 ],   [ -0.7, 1.1, 1.2 ], [ 1.3, 0.7, 0.0  ] ]
      ]

      # Should be 3x2x1
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ [ [ 5.51, 3.04, 4.3 ], [ 3.04, 6.31, 3.87 ] ] ]
    end

    it "should calculate a 4D convolution" do
      # 3x4x5x3
      a = NArray[
        [ [ [ 0.5, 0.4, 0.9 ], [ 0.1, 0.9, 0.8 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ],
          [ [ 0.0, 0.4, 0.0 ], [ 0.2, 0.3, 0.8 ], [ 0.6, 0.3, 0.2 ], [ 0.7, 0.4, 0.3 ] ],
          [ [ 0.3, 0.3, 0.1 ], [ 0.6, 0.9, 0.4 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ],
          [ [ 0.0, 0.4, 0.0 ], [ 0.2, 0.3, 0.8 ], [ 0.6, 0.3, 0.2 ], [ 0.7, 0.4, 0.3 ] ],
          [ [ 0.3, 0.3, 0.1 ], [ 0.6, 0.9, 0.4 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ] ],
        [ [ [ 0.5, 0.4, 0.9 ], [ 0.1, 0.9, 0.8 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ],
          [ [ 0.0, 0.4, 0.0 ], [ 0.2, 0.3, 0.8 ], [ 0.6, 0.3, 0.2 ], [ 0.7, 0.4, 0.3 ] ],
          [ [ 0.3, 0.3, 0.1 ], [ 0.6, 0.9, 0.4 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ],
          [ [ 0.0, 0.4, 0.0 ], [ 0.2, 0.3, 0.8 ], [ 0.6, 0.3, 0.2 ], [ 0.7, 0.4, 0.3 ] ],
          [ [ 0.3, 0.3, 0.1 ], [ 0.6, 0.9, 0.4 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ] ],
        [ [ [ 0.5, 0.4, 0.9 ], [ 0.1, 0.9, 0.8 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ],
          [ [ 0.0, 0.4, 0.0 ], [ 0.2, 0.3, 0.8 ], [ 0.6, 0.3, 0.2 ], [ 0.7, 0.4, 0.3 ] ],
          [ [ 0.3, 0.3, 0.1 ], [ 0.6, 0.9, 0.4 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ],
          [ [ 0.0, 0.4, 0.0 ], [ 0.2, 0.3, 0.8 ], [ 0.6, 0.3, 0.2 ], [ 0.7, 0.4, 0.3 ] ],
          [ [ 0.3, 0.3, 0.1 ], [ 0.6, 0.9, 0.4 ], [ 0.4, 0.0, 0.1 ], [ 0.8, 0.3, 0.4 ] ] ] ]

      # 2x3x3x2
      b = NArray[ [
        [ [ 1.1, 0.6 ], [ 1.2, 0.6 ], [ 0.8, 0.1 ] ], [ [ -0.4, 0.8 ], [ 0.5, 0.4 ], [ 1.2, 0.2 ] ],
        [ [ 0.8, 0.2 ], [ 0.5, 0.0 ], [ 1.4, 1.3 ] ] ],
        [ [ [ 1.1, 0.6 ], [ 1.2, 0.6 ], [ 0.8, 0.1 ] ], [ [ -0.4, 0.8 ], [ 0.5, 0.4 ], [ 1.2, 0.2 ] ],
        [ [ 0.8, 0.2 ], [ 0.5, 0.0 ], [ 1.4, 1.3 ] ] ] ]

      # Should be 2x2x3x2
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[
        [ [ [ 8.5, 8.2 ], [ 11.34, 9.68 ] ], [ [ 7.68, 6.56 ], [ 11.24, 7.16 ] ], [ [ 9.14, 6.54 ], [ 12.44, 9.2 ] ] ],
        [ [ [ 8.5, 8.2 ], [ 11.34, 9.68 ] ], [ [ 7.68, 6.56 ], [ 11.24, 7.16 ] ], [ [ 9.14, 6.54 ], [ 12.44, 9.2 ] ] ]
      ]
    end

    describe "compared with #convolve" do
      it "should produce same results for 1D arrays " do
        (1..30).each do |signal_length|
          (1..signal_length).each do |kernel_length|
            signal = NArray.sfloat(signal_length).random()
            kernel = NArray.sfloat(kernel_length).random()
            expect_result = Convolver.convolve( signal, kernel )
            got_result = Convolver.convolve_fftw3( signal, kernel )
            got_result.should be_narray_like expect_result
          end
        end
      end

      it "should produce same results for 2D arrays " do
        (3..10).each do |signal_x|
          (signal_x-2..signal_x+2).each do |signal_y|
            (1..signal_x).each do |kernel_x|
              (1..signal_y).each do |kernel_y|
                signal = NArray.sfloat(signal_x,signal_y).random()
                kernel = NArray.sfloat(kernel_x,kernel_y).random()
                expect_result = Convolver.convolve( signal, kernel )
                got_result = Convolver.convolve_fftw3( signal, kernel )
                got_result.should be_narray_like expect_result
              end
            end
          end
        end
      end

      it "should produce same results for 3D arrays " do
        (3..5).each do |signal_x|
          (signal_x-2..signal_x+2).each do |signal_y|
            (signal_x-2..signal_x+2).each do |signal_z|
              (1..signal_x).each do |kernel_x|
                (1..signal_y).each do |kernel_y|
                  (1..signal_z).each do |kernel_z|
                    signal = NArray.sfloat(signal_x,signal_y,signal_z).random()
                    kernel = NArray.sfloat(kernel_x,kernel_y,kernel_z).random()
                    expect_result = Convolver.convolve( signal, kernel )
                    got_result = Convolver.convolve_fftw3( signal, kernel )
                    got_result.should be_narray_like expect_result
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end
