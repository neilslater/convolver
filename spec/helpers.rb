# frozen_string_literal: true

# convolver/spec/helpers.rb
unless ENV['CONVOLVER_DISABLE_SIMPLECOV']
  require 'simplecov'
  SimpleCov.start do
    enable_coverage :branch
  end
end

require 'convolver'

# Keep the large numerical fixtures readable while constructing the new array type.
NArray = Numo::SFloat

class << NArray
  def sfloat(*shape)
    new(*shape)
  end
end

NArray.alias_method :random, :rand

require 'convolution_fixtures'

# Matcher compares Numo::NArray values numerically.
RSpec::Matchers.define :be_narray_like do |expected_narray, mse = 1e-9|
  match do |given|
    @error = nil
    if given.is_a?(Numo::NArray)
      @error = 'Shapes are different.' if given.shape != expected_narray.shape
    else
      @error = 'Wrong class.'
    end

    unless @error
      d = given - expected_narray
      difference = (d * d).sum / d.size
      @error = "Numerical difference with mean square error #{difference}" if difference > mse
    end
    @given = given.clone

    @expected = expected_narray.clone if @error

    !@error
  end

  failure_message do
    "Numo::NArray does not match supplied example. #{@error}
    Expected: #{@expected.inspect}
    Got: #{@given.inspect}"
  end

  failure_message_when_negated do
    "Numo::NArray is too close to unwanted example.
    Unwanted: #{@given.inspect}"
  end

  description do |_given, _expected|
    'numerically very close to example'
  end
end
