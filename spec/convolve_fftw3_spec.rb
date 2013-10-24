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

    describe "compared with #convolve" do
      it "should produce same results for 1D arrays " do
        (1..30).each do |signal_length|
          (1..signal_length).each do |kernel_length|
            signal = NArray.float(signal_length).random()
            kernel = NArray.float(kernel_length).random()
            expect_result = Convolver.convolve( signal, kernel )
            got_result = Convolver.convolve_fftw3( signal, kernel )
            got_result.should be_narray_like expect_result
          end
        end
      end
    end
  end
end
