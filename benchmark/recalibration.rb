# frozen_string_literal: true

require 'json'
require 'convolver'
require_relative 'recalibration/environment'
require_relative 'recalibration/cases'
require_relative 'recalibration/study_case'

suite = ENV.fetch('RECALIBRATION_SUITE', 'training')
definitions = suite == 'holdout' ? Recalibration::HOLDOUT : Recalibration::TRAINING
rows = [Numo::SFloat, Numo::DFloat].product(%i[convolve correlate]).flat_map do |type, operation|
  definitions.map { |definition| Recalibration::StudyCase.new(definition, type, operation).run }
end
conversion_names = suite == 'holdout' ? %w[short-wide reflect-fft wrap-moved] : %w[1d8 1d256 reflect]
conversion_cases = definitions.select { |definition| conversion_names.include?(definition.first) }
conversion_cases.each do |definition|
  [Numo::Int16, Numo::Int32, Numo::DFloat].each do |type|
    rows << Recalibration::StudyCase.new(definition, type, :convolve, view: type == Numo::DFloat).run
  end
end

puts JSON.pretty_generate(environment: Recalibration::Environment.capture, suite: suite,
                          sampling: '3 warmup, 3 pilot, 7 interleaved batches targeting 6 ms; 3..400 calls; normal GC',
                          rows: rows)
