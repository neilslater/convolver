# frozen_string_literal: true

require 'mkmf'
require 'numo/narray/alt'

$LOAD_PATH.each do |load_path|
  next unless File.exist?(File.join(load_path, 'numo/numo/narray.h'))

  $INCFLAGS = "-I#{File.join(load_path, 'numo')} #{$INCFLAGS}"
  break
end

abort 'numo/narray.h not found' unless have_header('numo/narray.h')

if RUBY_PLATFORM.match?(/mswin|cygwin|mingw/)
  $LOAD_PATH.each do |load_path|
    next unless File.exist?(File.join(load_path, 'numo/narray/libnarray.a'))

    $LDFLAGS = "-L#{File.join(load_path, 'numo/narray')} #{$LDFLAGS}"
    break
  end
  abort 'libnarray.a not found' unless have_library('narray', 'nary_new')
end

if RUBY_PLATFORM.include?('darwin') && Gem::Version.new(RUBY_VERSION) >= Gem::Version.new('3.1') &&
   try_link('int main(void) { return 0; }', '-Wl,-undefined,dynamic_lookup')
  $LDFLAGS << ' -Wl,-undefined,dynamic_lookup'
end

case ENV.fetch('CONVOLVER_NATIVE_MODE', 'release')
when 'release'
  $CFLAGS << ' -O3 -funroll-loops'
when 'lint'
  $CFLAGS << ' -std=gnu2x -O0 -g -Wall -Wextra -Wpedantic -Wformat=2 -Werror'
  if RbConfig::CONFIG.fetch('CC').match?(/clang/) || RbConfig::CONFIG.fetch('host_os').match?(/darwin/)
    $CFLAGS << ' -Wno-c23-extensions -Wno-strict-prototypes -Wno-unused-parameter'
    $CFLAGS << ' -Wno-default-const-init-field-unsafe'
  end
when 'coverage'
  $CFLAGS << ' -O0 -g --coverage'
  $LDFLAGS << ' --coverage'
when 'sanitize'
  sanitizer_flags = ' -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
  $CFLAGS << sanitizer_flags
  $LDFLAGS << ' -fsanitize=address,undefined'
else
  abort "Unknown CONVOLVER_NATIVE_MODE: #{ENV.fetch('CONVOLVER_NATIVE_MODE', nil)}"
end

create_makefile('convolver/convolver')
