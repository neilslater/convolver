# frozen_string_literal: true

module DtypeFixtures
  INPUTS = {
    Numo::SFloat => Numo::SFloat, Numo::DFloat => Numo::DFloat,
    Numo::Int8 => Numo::SFloat, Numo::UInt8 => Numo::SFloat,
    Numo::Int16 => Numo::SFloat, Numo::UInt16 => Numo::SFloat,
    Numo::Int32 => Numo::DFloat, Numo::UInt32 => Numo::DFloat,
    Numo::Int64 => Numo::DFloat, Numo::UInt64 => Numo::DFloat
  }.freeze
  OPTIONS = [{}, { dtype: nil }, { dtype: Numo::SFloat }, { dtype: Numo::DFloat }].freeze
  MODES = [{ mode: :valid }, { mode: :full, fill_value: 0.125 },
           *%i[constant nearest reflect mirror wrap].map { |boundary| { mode: :same, boundary:, origin: -1 } }].freeze

  def expected_dtype(first, second, options)
    options[:dtype] || ([INPUTS.fetch(first), INPUTS.fetch(second)].include?(Numo::DFloat) ? Numo::DFloat : Numo::SFloat)
  end

  def expect_dtype_result(result, dtype, values)
    expect(result).to be_an_instance_of(dtype)
    expect(result).to be_narray_like(Numo::DFloat.cast(values), 1e-20)
  end
end
