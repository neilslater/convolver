# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'English'
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

  desc 'Run the Ruby specs with AddressSanitizer and UBSan (Linux recommended)'
  task :sanitize do
    cc = RbConfig::CONFIG.fetch('CC')
    compiler_version = Open3.capture2e(*Shellwords.split(cc), '--version').first
    unless RUBY_PLATFORM.match?(/linux/) && compiler_version.match?(/gcc/i) && !compiler_version.match?(/clang/i)
      abort 'c:sanitize currently requires Linux and GCC'
    end

    libasan = Open3.capture2e(*Shellwords.split(cc), '-print-file-name=libasan.so').first.strip
    abort 'GCC could not locate libasan.so' if libasan.empty? || libasan == 'libasan.so'

    rebuild_and_test_native.call('sanitize', test: false)
    sanitizer_env = { 'ASAN_OPTIONS' => 'detect_leaks=0', 'LD_PRELOAD' => libasan }
    rspec_command = [RbConfig.ruby, '-S', 'bundle', 'exec', 'rspec']
    without_simplecov = { 'CONVOLVER_DISABLE_SIMPLECOV' => '1' }
    probes = [
      ['Ruby startup', {}, [RbConfig.ruby, '-e', 'exit']],
      ['extension load', {}, [RbConfig.ruby, '-rbundler/setup', '-Ilib', '-e', 'require "convolver"']],
      ['basic convolution', {}, [
        RbConfig.ruby, '-rbundler/setup', '-Ilib', '-e',
        'require "convolver"; Convolver.convolve_basic(NArray[0.3, 0.4, 0.5], NArray[1.3, -0.5])'
      ]],
      ['all specs without SimpleCov', without_simplecov, [*rspec_command, 'spec']],
      *Dir['spec/*_spec.rb'].map do |spec_file|
        [File.basename(spec_file), without_simplecov, [*rspec_command, spec_file]]
      end,
      ['all specs with SimpleCov', {}, [RbConfig.ruby, '-S', 'bundle', 'exec', 'rake', 'test']]
    ]

    results = probes.map do |label, extra_env, command|
      puts "\n== Sanitizer probe: #{label} =="
      success = system(sanitizer_env.merge(extra_env), *command)
      [label, success, $CHILD_STATUS]
    end

    puts "\n== Sanitizer probe summary =="
    results.each do |label, success, status|
      detail = status&.signaled? ? "signal #{status.termsig}" : "exit #{status&.exitstatus}"
      puts format('%<label>-32s %<result>s (%<detail>s)',
                  label: label, result: success ? 'PASS' : 'FAIL', detail: detail)
    end
    abort 'One or more sanitizer probes failed' unless results.all? { |_label, success, _status| success }
  end
end
# rubocop:enable Metrics/BlockLength
