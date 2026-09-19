# frozen_string_literal: true

namespace :docs do
  desc 'Reject YARD warnings and undocumented public API objects'
  task :check do
    require 'yard'

    stats = YARD::CLI::Stats.new
    stats.run('--no-yardopts', '--no-document', '--no-cache', '--no-save',
              '--no-private', '--fail-on-warning', '--list-undoc')
    objects = stats.all_objects
    abort 'No public API objects found by YARD' if objects.empty?
    abort 'Public API documentation is incomplete' if objects.any? { |object| object.docstring.empty? }
  end
end
