# frozen_string_literal: true

# Fresh backing arrays make mutation checks independent of other examples.
module NativeViewFixtures
  module_function

  def backing_arrays(dtype = Numo::SFloat)
    [dtype[100, 2, -3, 4, 1, -2, 3, 2, 4, -1, 5, 3, -4], dtype[100, 1, -2, 3, 2, -1, 4, 2, -3]]
  end

  def views(layout, signal, kernel, kernel_length)
    case layout
    when :signal_offset then [signal[1..12], kernel[0...kernel_length]]
    when :kernel_offset then [signal[0..11], kernel[1..kernel_length]]
    when :both_offsets then [signal[1..12], kernel[1..kernel_length]]
    when :zero_offset then [signal[0..11], kernel[0...kernel_length]]
    when :reversed then [signal[1..12].reverse, kernel[1..kernel_length].reverse]
    when :indexed then indexed_views(signal, kernel, kernel_length)
    end
  end

  def indexed_views(signal, kernel, kernel_length)
    [signal[[4, 1, 7, 2, 9, 3, 8, 5, 12, 10]], kernel[(1..kernel_length).to_a.reverse]]
  end

  def shaped_views(layout, signal, kernel)
    case layout
    when :multidimensional then [signal[1..12].reshape(3, 4), kernel[1..8].reshape(2, 4)]
    when :transposed then [signal[1..12].reshape(3, 4).transpose, kernel[1..8].reshape(2, 4).transpose]
    when :scalar then [signal[1, false], kernel[2, false]]
    end
  end
end
