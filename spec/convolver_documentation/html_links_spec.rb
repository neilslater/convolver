# frozen_string_literal: true

require 'helpers'
require 'tmpdir'
require_relative '../../tasks/support/html_links'

describe ConvolverDocumentation::HtmlLinks do
  def check_pages(pages)
    Dir.mktmpdir('convolver-link-spec') do |directory|
      pages.each { |name, content| File.write(File.join(directory, name), content) }
      described_class.new(directory).check!
    end
  end

  it 'accepts files, named anchors and empty top-of-page fragments' do
    pages = { 'index.html' => '<a href="rules.html#input">Rules</a><a href="#">Top</a>',
              'rules.html' => '<h1 id="input">Inputs</h1><a href="index.html">Home</a>' }
    expect { check_pages(pages) }.not_to raise_error
  end

  it 'does not fetch external links' do
    expect { check_pages('index.html' => '<a href="https://example.invalid/">Remote</a>') }.not_to raise_error
  end

  it 'rejects an empty output directory instead of passing without checking pages' do
    expect { check_pages({}) }.to raise_error(ArgumentError, /index was not generated/)
  end

  it 'rejects a missing generated page' do
    expect { check_pages('index.html' => '<a href="rules.md">Rules</a>') }
      .to raise_error(ArgumentError, /Broken documentation link/)
  end

  it 'rejects a missing anchor even when its page exists' do
    expect { check_pages('index.html' => '<a href="#missing">Missing</a>') }
      .to raise_error(ArgumentError, /Missing documentation anchor/)
  end

  it 'checks percent-encoded anchors' do
    text = '<h1 id="some title">Title</h1><a href="#some%20title">Here</a>'
    expect { check_pages('index.html' => text) }.not_to raise_error
  end
end
