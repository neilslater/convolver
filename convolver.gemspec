# frozen_string_literal: true

require 'English'
lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'convolver/version'

Gem::Specification.new do |spec|
  spec.name          = 'convolver'
  spec.version       = Convolver::VERSION
  spec.authors       = ['Neil Slater']
  spec.email         = ['slobo777@gmail.com']
  spec.description   = 'Convolution for NArray'
  spec.summary       = 'Convolution for NArray'
  spec.homepage      = 'http://github.com/neilslater/convolver'
  spec.license       = 'MIT'
  spec.required_ruby_version = '>= 3.2'

  spec.add_dependency 'fftw3', '>= 0.3'
  spec.add_dependency 'narray', '>= 0.6.0.8'

  spec.files         = `git ls-files`.split($INPUT_RECORD_SEPARATOR)
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.extensions    = spec.files.grep(%r{/extconf\.rb$})
  spec.require_paths = ['lib']
  spec.metadata['rubygems_mfa_required'] = 'true'
end
