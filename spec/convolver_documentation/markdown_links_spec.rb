# frozen_string_literal: true

require 'helpers'
require_relative '../../tasks/support/markdown_links'

describe ConvolverDocumentation::MarkdownLinks do
  subject(:links) { described_class.new('/project', %w[README.md docs/rules.md docs/terminology.md]) }

  it 'resolves links relative to the page rather than the repository root' do
    result = links.convert('[Rules](rules.md) and [README](../README.md)', 'docs/terminology.md')
    expect(result).to eq('{file:docs/rules.md Rules} and {file:README.md README}')
  end

  it 'preserves remote links, images and same-page anchors' do
    text = '[Remote](https://example.com/rules.md) ![Image](plot.png) [Here](#here)'
    expect(links.convert(text, 'README.md')).to eq(text)
  end

  it 'preserves inline and fenced code containing Markdown examples' do
    text = "`[Rules](docs/rules.md)`\n```text\n[Rules](docs/rules.md)\n```\n"
    expect(links.convert(text, 'README.md')).to eq(text)
  end

  it 'preserves tilde-fenced examples' do
    text = "~~~text\n[Rules](docs/rules.md)\n~~~\n"
    expect(links.convert(text, 'README.md')).to eq(text)
  end

  it 'keeps explicit anchors on translated file references' do
    expect(links.convert('[Rules](docs/rules.md#input)', 'README.md')).to eq('{file:docs/rules.md#input Rules}')
  end

  it 'rejects links to pages that will not be generated' do
    expect { links.convert('[Private](agent/architecture.md)', 'docs/rules.md') }
      .to raise_error(ArgumentError, /Undocumented page/)
  end
end
