require 'helpers'

module Convolver
  def self.convolve_fftw3 orig_a, orig_b
    combined_size = orig_a.size + orig_b.size - 1
    left_pad_a = ( combined_size - orig_a.size + 1)/2
    mod_a = NArray.float(combined_size)
    mod_a[left_pad_a] = orig_a

    mod_b = NArray.float(combined_size)
    left_select_b = ( orig_b.size + 1 )/2
    right_select_b = orig_b.size - left_select_b
    mod_b[0] = orig_b[(0...left_select_b)].reverse
    mod_b[-right_select_b] = orig_b[-right_select_b..-1].reverse

    afft = FFTW3.fft(mod_a)
    bfft = FFTW3.fft(mod_b)
    cfft = afft * bfft

    puts "#{left_select_b}..#{(left_select_b + orig_a.size - orig_b.size)}"
    (FFTW3.ifft( cfft )/combined_size).real[left_select_b..(left_select_b + orig_a.size - orig_b.size)]
  end
end

describe Convolver do
  describe "#convolve_fftw3" do

    it "should work like the example in the README" do
      a = NArray[ 0.3, 0.4, 0.5 ]
      b = NArray[ 1.3, -0.5 ]
      c = Convolver.convolve_fftw3( a, b )
      c.should be_narray_like NArray[ 0.19, 0.27 ]
    end

    it "should convolve 1d arrays with a variety of odd or even lengths" do
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

  end
end
