# frozen_string_literal: true

require 'helpers'

describe Convolver do
  let(:size_max) { described_class.const_get(:BufferLimits)::SIZE_MAX }

  it 'keeps both planner primitives private' do
    expect(described_class.private_methods).to include(:fft_fast_shape, :fft_shape_cost)
  end

  [8, 14, 31, 64, 128].each do |maximum|
    it "matches independent smooth-number enumeration up to #{maximum}", :aggregate_failures do
      smooth_candidates(maximum).each do |input, expected|
        expect(fast_shape(input, maximum)).to eq(expected)
      end
    end
  end

  it 'accepts a large power of two without overflowing the next multiplication' do
    power = (size_max + 1) / 2
    expect(fast_shape([power], size_max)).to eq [power]
  end

  it 'returns no optional candidate when none fits near the native limit' do
    expect(fast_shape([size_max - 1], size_max)).to be_nil
  end

  it 'returns no candidate when a dimension exceeds the supplied limit' do
    expect(fast_shape([9, 2], 8)).to be_nil
  end

  it 'leaves frozen input dimensions untouched', :aggregate_failures do
    shape = [11, 13].freeze
    expect(fast_shape(shape, 100)).to eq [12, 16]
    expect(shape).to eq [11, 13]
  end

  %i[fft_fast_shape fft_shape_cost].each do |method|
    it "rejects non-array shapes in #{method}" do
      arguments = method == :fft_fast_shape ? [nil, size_max] : [nil]
      expect { described_class.send(method, *arguments) }.to raise_error(TypeError)
    end

    [[], [1] * 17, [0], [-1], [1.5], [nil]].each do |shape|
      it "rejects invalid dimensions #{shape.inspect} in #{method}" do
        arguments = method == :fft_fast_shape ? [shape, size_max] : [shape]
        expect { described_class.send(method, *arguments) }.to raise_error(ArgumentError)
      end
    end

    it "rejects values wider than size_t in #{method}" do
      arguments = method == :fft_fast_shape ? [[size_max + 1], size_max] : [[size_max + 1]]
      expect { described_class.send(method, *arguments) }.to raise_error(RangeError)
    end
  end

  [0, -1, 4.5, nil].each do |maximum|
    it "rejects an invalid candidate size limit #{maximum.inspect}" do
      expect { fast_shape([2], maximum) }.to raise_error(ArgumentError)
    end
  end

  it 'rejects a size limit wider than size_t' do
    expect { fast_shape([2], size_max + 1) }.to raise_error(RangeError)
  end

  it 'bounds factorization even when the private scoring primitive is called directly' do
    expect { described_class.send(:fft_shape_cost, [134_217_728]) }.to raise_error(RangeError, /scoring axis/)
  end

  it 'checks candidate products before scoring' do
    expect { described_class.send(:fft_shape_cost, [65_536] * 4) }.to raise_error(RangeError, /scoring size/)
  end

  it 'scores repeated large prime factors with their multiplicity' do
    expected = 121 * (Math.log2(121) + (3 * (Math.log2(11) - 3)))
    expect(described_class.send(:fft_shape_cost, [121])).to be_within(1e-10).of(expected)
  end

  def fast_shape(shape, maximum)
    described_class.send(:fft_fast_shape, shape, maximum)
  end

  def smooth_candidates(maximum)
    smooth = (1..maximum).select { |value| smooth?(value) }
    (1..maximum).flat_map do |target|
      even = smooth.find { |value| value >= target && value.even? }
      any = smooth.find { |value| value >= target }
      [[[target], optional_shape(even)], [[target, 2], optional_shape(any, 2)]]
    end
  end

  def optional_shape(first, *rest)
    [first, *rest] if first
  end

  def smooth?(value)
    [2, 3, 5].each { |factor| value /= factor while (value % factor).zero? }
    value == 1
  end
end
