# frozen_string_literal: true

require 'json'
require 'rbconfig'
require 'convolver'
require_relative 'selection_cases'
require_relative 'selection_case'

rows = [Numo::SFloat, Numo::DFloat].product(%i[convolve correlate]).flat_map do |type, operation|
  SelectionBenchmark::CASES.map { |definition| SelectionBenchmark::Case.new(definition, type, operation).run }
end
conversion_cases = SelectionBenchmark::CASES.select { |definition| %w[1d8 1d256 reflect].include?(definition.first) }
[Numo::Int16, Numo::Int32, Numo::DFloat].each do |type|
  conversion_cases.each do |definition|
    rows << SelectionBenchmark::Case.new(definition, type, :convolve, view: type == Numo::DFloat).run
  end
end

puts JSON.pretty_generate(
  ruby: RUBY_DESCRIPTION, platform: RUBY_PLATFORM, compiler: RbConfig::CONFIG['CC_VERSION_MESSAGE'],
  dependencies: %w[numo-narray-alt numo-pocketfft].to_h { |name| [name, Gem.loaded_specs.fetch(name).version.to_s] },
  sampling: '3 warmup calls, 3 pilot calls, 5 interleaved batches targeting 4 ms, 3..200 calls/batch; normal GC',
  rows: rows
)
