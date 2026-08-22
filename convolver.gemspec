# frozen_string_literal: true

lib = File.expand_path('lib', __dir__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'convolver/version'

Gem::Specification.new do |spec|
  spec.name          = 'convolver'
  spec.version       = Convolver::VERSION
  spec.authors       = ['Neil Slater']
  spec.email         = ['slobo777@gmail.com']
  spec.description   = 'Fast mathematical convolution and cross-correlation for multidimensional ' \
                       'Numo::NArray values, with configurable output and boundary modes.'
  spec.summary       = 'Fast convolution and cross-correlation for Numo::NArray'
  spec.homepage      = 'https://github.com/neilslater/convolver'
  spec.license       = 'MIT'
  spec.required_ruby_version = '>= 3.3'

  spec.add_dependency 'numo-narray-alt', '>= 0.9.9', '< 0.11'
  spec.add_dependency 'numo-pocketfft', '>= 0.6', '< 0.8'

  spec.files         = Dir['CHANGELOG.md', 'LICENSE.txt', 'README.md', 'lib/**/*.rb', 'ext/**/*.{c,h,rb}']
  spec.extensions    = spec.files.grep(%r{/extconf\.rb$})
  spec.require_paths = ['lib']
  spec.metadata['rubygems_mfa_required'] = 'true'
  spec.metadata['source_code_uri'] = spec.homepage
end
