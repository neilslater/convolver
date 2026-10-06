# frozen_string_literal: true

require 'pathname'
require 'uri'

module ConvolverDocumentation
  # Adapts inline local Markdown links in render copies, never in source files.
  class MarkdownLinks
    LINKS = /
      (?<code>^\ {0,3}(?<fence>`{3,}|~{3,})[^\n]*\n.*?^\ {0,3}\k<fence>[\t\ ]*$|`[^`\n]*`)
      | (?<!!)\[(?<label>[^\]]+)\]\((?<target>[^)\s]+)\)
    /mx

    def initialize(root, pages)
      @root = Pathname.new(root)
      @pages = pages
    end

    def convert(text, filename)
      text.gsub(LINKS) do |original|
        match = Regexp.last_match
        match[:code] ? original : convert_link(match, filename)
      end
    end

    private

    def convert_link(match, filename)
      target = URI.parse(match[:target])
      return match[0] if target.scheme || target.host || !target.path.end_with?('.md')

      path = local_path(filename, target.path)
      raise ArgumentError, "Undocumented page in #{filename}: #{path}" unless @pages.include?(path)

      anchor = "##{target.fragment}" if target.fragment
      "{file:#{path}#{anchor} #{match[:label]}}"
    end

    def local_path(filename, target)
      @root.join(filename).dirname.join(target).cleanpath.relative_path_from(@root).to_s
    end
  end
end
