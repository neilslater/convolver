# frozen_string_literal: true

# Literal asymmetric fixtures for mathematical convolution.
module ConvolutionFixtures
  module_function

  def one_dimensional_signal
    NArray[1, 2, 4]
  end

  def one_dimensional_kernel
    NArray[3, 5]
  end

  def one_dimensional_valid
    NArray[11, 22]
  end

  def one_dimensional_same
    NArray[11, 22, 20]
  end

  def one_dimensional_full
    NArray[3, 11, 22, 20]
  end

  def two_dimensional_signal
    NArray[[0.3, 0.4, 0.5], [0.6, 0.8, 0.2], [0.9, 1.0, 0.1]]
  end

  def two_dimensional_kernel
    NArray[[1.2, -0.5], [0.5, -1.3]]
  end

  def two_dimensional_valid
    NArray[[0.47, -0.43], [0.37, -1.32]]
  end
end
