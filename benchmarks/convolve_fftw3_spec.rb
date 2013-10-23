require 'fftw3'

# This is a Ruby prototype for convolving larger kernels using fftw3.
module Convolver
  def self.convolve_fftw3 orig_a, orig_b
    combined_size = orig_a.size + orig_b.size - 1
    output_size = orig_a.size - orig_b.size + 1
    output_offset = orig_b.size - 1

    left_pad_a = ( combined_size - orig_a.size + 1)/2
    mod_a = NArray.float(combined_size)
    mod_a[left_pad_a] = orig_a

    mod_b = NArray.float(combined_size)
    left_select_b = ( orig_b.size + 1 )/2

    right_select_b = orig_b.size - left_select_b
    b_rev = orig_b.reverse
    mod_b[0] = orig_b[(0...left_select_b)].reverse
    mod_b[-right_select_b] = orig_b[-right_select_b..-1].reverse

    afft = FFTW3.fft(mod_a)
    bfft = FFTW3.fft(mod_b)
    cfft = afft * bfft

    # puts " #{output_offset}..#{output_offset + output_size - 1}"
    (FFTW3.ifft( cfft )/combined_size).real[output_offset...(output_offset + output_size)]
  end
end



# Matcher compares NArrays numerically
RSpec::Matchers.define :be_narray_like do |expected_narray|
  match do |given|
    @error = nil
    if ! given.is_a?(NArray)
      @error = "Wrong class."
    elsif given.shape != expected_narray.shape
      @error = "Shapes are different."
    else
      d = given - expected_narray
      difference =  ( d * d ).sum / d.size
      if difference > 1e-10
        @error = "Numerical difference with mean square error #{difference}"
      end
    end
    @given = given.clone

    if @error
      @expected = expected_narray.clone
    end

    ! @error
  end

  failure_message_for_should do
    "NArray does not match supplied example. #{@error}
    Expected: #{@expected.inspect}
    Got: #{@given.inspect}"
  end

  failure_message_for_should_not do
    "NArray is too close to unwanted example.
    Got: #{@given.inspect}"
  end

  description do |given, expected|
    "numerically very close to example"
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

    it "should convolve 1d arrays with a variety of signal and kernel lengths" do
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

  end
end
