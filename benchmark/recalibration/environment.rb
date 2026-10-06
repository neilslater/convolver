# frozen_string_literal: true

require 'open3'
require 'shellwords'
require 'rbconfig'

module Recalibration
  # Records host/build provenance without requiring a platform-specific gem.
  module Environment
    def self.capture
      { ruby: RUBY_DESCRIPTION, platform: RUBY_PLATFORM, cpu: cpu, os: command('uname', '-a'),
        compiler: RbConfig::CONFIG['CC_VERSION_MESSAGE'], compiler_macros: compiler_macros,
        extension_flags: extension_flags, commit: command('git', 'rev-parse', 'HEAD'),
        dependencies: %w[numo-narray-alt numo-pocketfft].to_h do |name|
          [name, Gem.loaded_specs.fetch(name).version.to_s]
        end }
    end

    def self.cpu
      return command('sysctl', '-n', 'machdep.cpu.brand_string') if RUBY_PLATFORM.include?('darwin')

      File.readlines('/proc/cpuinfo').grep(/^(model name|Hardware|CPU implementer|CPU part|Features|flags)\s*:/).uniq
    end

    def self.compiler_macros
      compiler = Shellwords.split(RbConfig::CONFIG.fetch('CC'))
      output = Open3.capture2e(*compiler, '-dM', '-E', '-x', 'c', '-', stdin_data: '').first
      output.lines.grep(/__(SSE|SSE2|aarch64|ARM_NEON|x86_64)__/).map(&:strip)
    end

    def self.extension_flags
      path = Dir['tmp/**/convolver/**/Makefile'].find { |name| name.include?(RUBY_VERSION) }
      path && File.readlines(path).grep(/^(CC|CFLAGS|CPPFLAGS|LDFLAGS)\s*=/).map(&:strip)
    end

    def self.command(*)
      Open3.capture2e(*).first.strip
    end
  end
end
