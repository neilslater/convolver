require "bundler/gem_tasks"
require "rspec/core/rake_task"
require 'rake/extensiontask'

desc "Convolver unit tests"
RSpec::Core::RakeTask.new(:test) do |t|
  t.pattern = "spec/*_spec.rb"
  t.verbose = true
end

gemspec = Gem::Specification.load('convolver.gemspec')
Rake::ExtensionTask.new do |ext|
  ext.name = 'convolver'
  ext.source_pattern = "*.{c,h}"
  ext.ext_dir = 'ext/convolver'
  ext.lib_dir = 'lib/convolver'
  ext.gem_spec = gemspec
end

task :default => [:compile, :test]