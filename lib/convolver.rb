require 'narray'
require "convolver/convolver"
require "convolver/version"

module Convolver
  # Calculates float convolution of an array with a kernel
  # @param [NArray] a outer array
  # @param [NArray] b kernel
  # @return [NArray] result of convolving a with b
  # @!parse def self.convolve(a,b); end
end
