# frozen_string_literal: true

require 'cgi'
require 'uri'

module ConvolverDocumentation
  # Checks generated local destinations without depending on network availability.
  class HtmlLinks
    def initialize(directory)
      @directory = directory
    end

    def check!
      unless File.file?(File.join(@directory, 'index.html'))
        raise ArgumentError, 'Documentation index was not generated'
      end

      Dir.glob(File.join(@directory, '**/*.html')).each do |page|
        File.read(page).scan(/<a\b[^>]*\bhref=(["'])(.*?)\1/m).map(&:last).each do |href|
          check_link(page, CGI.unescapeHTML(href))
        end
      end
    end

    private

    def check_link(page, href)
      uri = URI.parse(href)
      return if uri.scheme || uri.host

      target = uri.path.empty? ? page : File.expand_path(URI::DEFAULT_PARSER.unescape(uri.path), File.dirname(page))
      raise ArgumentError, "Broken documentation link in #{page}: #{href}" unless File.file?(target)

      check_anchor(target, uri.fragment) if uri.fragment && File.extname(target) == '.html'
    end

    def check_anchor(target, fragment)
      return if fragment.empty?

      anchors = File.read(target).scan(/\b(?:id|name)=(["'])(.*?)\1/m).map(&:last)
      return if anchors.include?(URI::DEFAULT_PARSER.unescape(fragment))

      raise ArgumentError, "Missing documentation anchor: #{target}##{fragment}"
    end
  end
end
