# frozen_string_literal: true

require 'helpers'

describe RSpec::Matchers, '#be_narray_like' do
  it 'accepts exact finite matches' do
    expect(NArray[-2, 0, 3]).to be_narray_like NArray[-2, 0, 3]
  end

  it 'rejects values of the wrong class' do
    matcher = be_narray_like(NArray[0])
    expect(matcher.matches?([0])).to be false
  end

  it 'rejects different shapes with the same elements' do
    expect(NArray[[1, 2]]).not_to be_narray_like NArray[1, 2]
  end

  it 'accepts a mean square error just inside the tolerance' do
    expect(NArray[0.499]).to be_narray_like(NArray[0], 0.25)
  end

  it 'accepts a mean square error equal to the tolerance' do
    expect(NArray[0.5]).to be_narray_like(NArray[0], 0.25)
  end

  it 'rejects a mean square error just outside the tolerance' do
    expect(NArray[0.501]).not_to be_narray_like(NArray[0], 0.25)
  end

  it 'rejects a single bad finite element when the mean square error exceeds the tolerance' do
    expect(NArray[0, 0, 1, 0]).not_to be_narray_like(NArray.zeros(4), 0.2)
  end

  [Float::NAN, Float::INFINITY, -Float::INFINITY].each do |value|
    it "rejects #{value} against finite expected values" do
      expect(NArray[0, value, 0]).not_to be_narray_like NArray.zeros(3)
    end

    it "rejects finite values against expected #{value}" do
      expect(NArray.zeros(3)).not_to be_narray_like NArray[0, value, 0]
    end

    it "rejects matching #{value} values" do
      expect(NArray[value]).not_to be_narray_like NArray[value]
    end
  end

  it 'rejects infinities with opposite signs' do
    expect(NArray[Float::INFINITY]).not_to be_narray_like NArray[-Float::INFINITY]
  end

  it 'rejects overflow in the mean square error of finite values' do
    expect(NArray[1e20]).not_to be_narray_like(NArray[0], Float::INFINITY)
  end

  it 'reports the non-finite error statistic on failure' do
    matcher = be_narray_like(NArray[0])
    matcher.matches?(NArray[Float::NAN])
    expect(matcher.failure_message).to include('mean square error NaN')
  end
end
