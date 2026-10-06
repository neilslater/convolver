# frozen_string_literal: true

require 'helpers'
require 'yard'

describe Convolver do
  let(:public_module) do
    YARD::Registry.clear
    YARD.parse(['lib/**/*.rb', 'ext/**/*.c'])
    YARD::Registry.at('Convolver')
  end
  let(:public_methods) do
    public_module.meths.select do |method|
      method.scope == :class && method.visibility == :public && !method.has_tag?(:private)
    end
  end

  it 'preserves the public overview when implementation files reopen the module' do
    expect(public_module.docstring.to_s).to start_with('Combine numeric sequences or grids')
  end

  it 'keeps exactly the runtime public methods in the reference' do
    expect(public_methods.map(&:name)).to match_array(described_class.singleton_methods(false))
  end

  it 'documents each parameter once for every public method' do
    expected = %w[signal kernel mode boundary fill_value origin dtype]
    expect(public_methods.map { |method| method.tags(:param).map(&:name).sort }).to all(eq(expected.sort))
  end

  it 'keeps the true omitted-fill default in each signature' do
    defaults = public_methods.map { |method| method.parameters.assoc('fill_value:').last }
    expect(defaults).to all(eq('UNSPECIFIED_FILL'))
  end

  it 'does not inherit FFT-specific errors for direct calculation or estimation' do
    direct = public_methods.select { |method| method.name.to_s.match?(/_basic(?:_time)?\z/) }
    descriptions = direct.flat_map { |method| method.tags(:raise).map(&:text) }
    expect(descriptions).not_to include(a_string_including('FFT'))
  end

  it 'provides a worked example for every public method' do
    expect(public_methods.map { |method| method.tags(:example) }).to all(satisfy { |examples| !examples.empty? })
  end
end
