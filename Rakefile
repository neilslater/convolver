# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'fileutils'
require 'open3'
require 'rspec/core/rake_task'
require 'rake/extensiontask'
require 'rbconfig'
require 'shellwords'

desc 'Convolver unit tests'
RSpec::Core::RakeTask.new(:test) do |t|
  t.pattern = 'spec/*_spec.rb'
  t.verbose = true
end

gemspec = Gem::Specification.load('convolver.gemspec')
Rake::ExtensionTask.new do |ext|
  ext.name = 'convolver'
  ext.source_pattern = '*.{c,h}'
  ext.ext_dir = 'ext/convolver'
  ext.lib_dir = 'lib/convolver'
  ext.gem_spec = gemspec
end

task default: %i[compile test]

rebuild_and_test_native = lambda do |mode, test: true|
  tasks = %w[clean compile]
  tasks << 'test' if test
  sh({ 'CONVOLVER_NATIVE_MODE' => mode }, RbConfig.ruby, '-S', 'bundle', 'exec', 'rake', *tasks)
end

# rubocop:disable Metrics/BlockLength
namespace :c do
  desc 'Compile the C extension with strict warnings'
  task :lint do
    rebuild_and_test_native.call('lint', test: false)
  end

  desc 'Measure C coverage while running the Ruby specs (requires GCC and gcovr)'
  task :coverage do
    cc = RbConfig::CONFIG.fetch('CC')
    compiler_version = Open3.capture2e(*Shellwords.split(cc), '--version').first
    unless compiler_version.match?(/gcc/i) && !compiler_version.match?(/clang/i)
      abort "c:coverage requires a GCC Ruby build (current compiler: #{cc})"
    end
    abort 'c:coverage requires gcovr on PATH' unless system('gcovr', '--version', out: File::NULL)

    rebuild_and_test_native.call('coverage')
    FileUtils.mkdir_p('coverage/c')
    sh 'gcovr', '--root', '.', '--filter', 'ext/convolver/', '--html-details',
       'coverage/c/index.html', '--xml', 'coverage/c/cobertura.xml', '--txt',
       'coverage/c/summary.txt', '--print-summary'
  end

  desc 'Run the Ruby specs with AddressSanitizer and UBSan (requires Linux and GCC)'
  task :sanitize do
    cc = RbConfig::CONFIG.fetch('CC')
    compiler_version = Open3.capture2e(*Shellwords.split(cc), '--version').first
    unless RUBY_PLATFORM.match?(/linux/) && compiler_version.match?(/gcc/i) && !compiler_version.match?(/clang/i)
      abort 'c:sanitize currently requires Linux and GCC'
    end

    libasan = Open3.capture2e(*Shellwords.split(cc), '-print-file-name=libasan.so').first.strip
    abort 'GCC could not locate libasan.so' if libasan.empty? || libasan == 'libasan.so'

    rebuild_and_test_native.call('sanitize', test: false)
    sanitizer_env = {
      'ASAN_OPTIONS' => 'detect_leaks=0',
      'CONVOLVER_DISABLE_SIMPLECOV' => '1',
      'LD_PRELOAD' => libasan
    }
    sh(sanitizer_env, RbConfig.ruby, '-S', 'bundle', 'exec', 'rake', 'test')
  end
end
# rubocop:enable Metrics/BlockLength
