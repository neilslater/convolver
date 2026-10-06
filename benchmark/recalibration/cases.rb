# frozen_string_literal: true

require_relative '../selection_cases'

module Recalibration
  TRAINING = [
    *SelectionBenchmark::CASES,
    ['long-fft', [16_384], [1024], {}],
    ['large-2d-fft', [128, 128], [33, 33], {}],
    ['large-3d-fft', [32, 32, 32], [15, 15, 15], {}]
  ].freeze

  # Declared before fitting; reserve these shapes and independent runs for validation.
  HOLDOUT = [
    ['short-wide', [511], [255], {}],
    ['awkward-wide', [1009], [127], {}],
    ['below-kernel-threshold', [2048], [63], {}],
    ['at-kernel-threshold', [2048], [64], {}],
    ['above-kernel-threshold', [2048], [65], {}],
    ['mid-1d', [8192], [192], {}],
    ['large-1d', [8192], [512], {}],
    ['narrow-2d', [3, 769], [2, 129], {}],
    ['wide-2d', [48, 80], [9, 13], {}],
    ['square-2d', [96, 96], [21, 21], {}],
    ['small-3d', [12, 14, 18], [3, 3, 9], {}],
    ['wide-3d', [24, 28, 20], [7, 9, 5], {}],
    ['large-3d', [28, 30, 32], [13, 13, 13], {}],
    ['reflect-fft', [3072], [255], { mode: :same, boundary: :reflect }],
    ['nearest-2d', [48, 64], [9, 7], { mode: :same, boundary: :nearest }],
    ['mirror-3d', [16, 18, 20], [3, 5, 7], { mode: :same, boundary: :mirror }],
    ['full-1d', [3072], [127], { mode: :full }],
    ['full-2d', [48, 48], [17, 17], { mode: :full }],
    ['wrap-1d', [2048], [31], { mode: :same, boundary: :wrap }],
    ['wrap-small', [19, 17], [3, 3], { mode: :same, boundary: :wrap }],
    ['wrap-collision-2d', [17, 19], [24, 23], { mode: :same, boundary: :wrap }],
    ['wrap-moved', [48, 47], [11, 13], { mode: :same, boundary: :wrap }],
    ['wrap-3d', [18, 20, 22], [7, 5, 3], { mode: :same, boundary: :wrap }],
    ['rank-four', [6, 8, 10, 12], [3, 3, 3, 3], {}]
  ].freeze
end
