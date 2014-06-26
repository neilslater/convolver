# convolver/spec/helpers.rb
require 'coveralls'
require 'mocha/api'

Coveralls.wear!

require 'convolver'

# Matcher compares NArrays numerically
RSpec::Matchers.define :be_narray_like do |expected_narray, mse = 1e-9|
  match do |given|
    @error = nil
    if ! given.is_a?(NArray)
      @error = "Wrong class."
    elsif given.shape != expected_narray.shape
      @error = "Shapes are different."
    else
      d = given - expected_narray
      difference =  ( d * d ).sum / d.size
      if difference > mse
        @error = "Numerical difference with mean square error #{difference}"
      end
    end
    @given = given.clone

    if @error
      @expected = expected_narray.clone
    end

    ! @error
  end

  failure_message do
    "NArray does not match supplied example. #{@error}
    Expected: #{@expected.inspect}
    Got: #{@given.inspect}"
  end

  failure_message_when_negated do
    "NArray is too close to unwanted example.
    Unwanted: #{@given.inspect}"
  end

  description do |given, expected|
    "numerically very close to example"
  end
end
