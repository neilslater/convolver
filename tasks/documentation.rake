# frozen_string_literal: true

namespace :docs do
  desc 'Build linked API and reference pages in doc/index.html'
  task :build do
    require_relative 'support/documentation_build'

    ConvolverDocumentation::Build.new(File.expand_path('..', __dir__)).run
  end

  desc 'Reject YARD warnings and undocumented public API objects'
  task check: :build do
    require 'yard'

    stats = YARD::CLI::Stats.new
    stats.run('--no-yardopts', '--no-document', '--no-cache', '--no-save',
              '--no-private', '--fail-on-warning', '--list-undoc')
    # Reusable YARD text macros are registry entries, not public Ruby objects.
    objects = stats.all_objects.reject { |object| object.type == :macro }
    abort 'No public API objects found by YARD' if objects.empty?
    abort 'Public API documentation is incomplete' if objects.any? { |object| object.docstring.empty? }
  end
end
