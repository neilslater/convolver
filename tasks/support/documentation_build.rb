# frozen_string_literal: true

require 'fileutils'
require 'tmpdir'
require 'yard'
require_relative 'markdown_links'
require_relative 'html_links'

module ConvolverDocumentation
  # Keeps GitHub/installable Markdown portable while producing linked YARD HTML.
  class Build
    PAGES = %w[README.md CHANGELOG.md docs/rules.md docs/terminology.md].freeze
    SOURCES = %w[lib/**/*.rb ext/**/*.c].freeze

    def initialize(root)
      @root = File.expand_path(root)
      @output = File.join(@root, 'doc')
    end

    def run
      Dir.mktmpdir('convolver-docs') do |directory|
        prepare(directory)
        render(directory)
      end
      HtmlLinks.new(@output).check!
    end

    private

    def prepare(directory)
      sources = Dir.chdir(@root) { Dir.glob(SOURCES) }
      (sources + PAGES).each do |filename|
        target = File.join(directory, filename)
        FileUtils.mkdir_p(File.dirname(target))
        FileUtils.cp(File.join(@root, filename), target)
      end
      convert_pages(directory)
    end

    def convert_pages(directory)
      links = MarkdownLinks.new(@root, PAGES)
      PAGES.each do |filename|
        target = File.join(directory, filename)
        File.write(target, links.convert(File.read(target), filename))
      end
    end

    def render(directory)
      FileUtils.rm_rf(@output)
      Dir.chdir(directory) do
        YARD::Registry.clear
        YARD::CLI::Yardoc.new.run('--no-yardopts', '--no-document', '--no-cache', '--no-save',
                                  '--no-private', '--fail-on-warning', '--readme', 'README.md',
                                  '--files', PAGES.drop(1).join(','), '--output-dir', @output, *SOURCES)
      end
    end
  end
end
