# frozen_string_literal: true

module SelectionBenchmark
  CASES = [
    ['scalar', [], [], {}],
    ['threshold999', [999], [3], {}],
    ['threshold1000', [1000], [3], {}],
    ['threshold1001', [1001], [3], {}],
    ['small-wide', [999], [500], {}],
    ['1d8', [4096], [8], {}],
    ['1d64', [4096], [64], {}],
    ['1d128', [4096], [128], {}],
    ['1d256', [4096], [256], {}],
    ['prime', [4093], [15], {}],
    ['2d5', [64, 64], [5, 5], {}],
    ['2d16', [64, 64], [16, 16], {}],
    ['2d32', [64, 64], [32, 32], {}],
    ['2d-prime', [97, 101], [7, 11], {}],
    ['3d3', [16, 16, 16], [3, 3, 3], {}],
    ['3d7', [16, 16, 16], [7, 7, 7], {}],
    ['3d-large', [32, 32, 32], [7, 7, 7], {}],
    ['same', [4096], [31], { mode: :same }],
    ['full', [4096], [31], { mode: :full }],
    ['reflect', [4096], [31], { mode: :same, boundary: :reflect, origin: 1 }],
    ['nearest', [2048], [17], { mode: :same, boundary: :nearest }],
    ['mirror', [2048], [17], { mode: :same, boundary: :mirror }],
    ['2d-reflect', [64, 64], [5, 5], { mode: :same, boundary: :reflect, origin: -1 }],
    ['wrap1d', [4096], [63], { mode: :same, boundary: :wrap }],
    ['wrap-odd', [63, 63], [5, 5], { mode: :same, boundary: :wrap }],
    ['wrap-axis', [64, 63], [17, 17], { mode: :same, boundary: :wrap }],
    ['wrap-collision', [32, 32], [48, 48], { mode: :same, boundary: :wrap }],
    ['wrap3d', [16, 18, 20], [5, 5, 5], { mode: :same, boundary: :wrap }],
    ['rank4', [8, 8, 8, 8], [3, 3, 3, 3], {}],
    ['rank5', [4, 4, 4, 4, 4], [2, 2, 2, 2, 2], {}],
    ['crossover1d', [769], [65], {}],
    ['crossover2d', [40, 48], [9, 9], {}],
    ['thin', [2, 1024], [1, 257], {}]
  ].freeze
end
