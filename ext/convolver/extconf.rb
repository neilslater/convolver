# frozen_string_literal: true

# ext/convolver/extconf.rb
require 'mkmf'
require 'rubygems'

# Following code stolen shamelessly from fftw3 gem:
narray_dir = begin
  File.dirname(Gem.find_files('narray.h').first)
rescue StandardError
  $sitearchdir
end
if /cygwin|mingw/ =~ RUBY_PLATFORM
  dir_config('narray', narray_dir, "#{narray_dir}/src")
else
  dir_config('narray', narray_dir, narray_dir)
end

unless have_header('narray.h') && have_header('narray_config.h')
  print <<-ERROR_MESSAGE
   ** configure error **
   Header narray.h or narray_config.h is not found. If you have these files in
   /narraydir/include, try the following:

   % ruby extconf.rb --with-narray-include=/narraydir/include

  ERROR_MESSAGE
  exit(-1)
end

# This also stolen from fftw3 gem (and not confirmed for Windows platforms - please let me know if it works!)
if /cygwin|mingw/ =~ RUBY_PLATFORM
  have_library('narray') || raise('ERROR: narray library is not found')
end

case ENV.fetch('CONVOLVER_NATIVE_MODE', 'release')
when 'release'
  $CFLAGS << ' -O3 -funroll-loops'
when 'lint'
  $CFLAGS << ' -std=gnu2x -O0 -g -Wall -Wextra -Wpedantic -Wformat=2 -Werror'
  # narray 0.6 owns the wrapped objects and exposes them through the legacy
  # Data_Get_Struct API, so Convolver cannot migrate those accesses to TypedData.
  $CFLAGS << ' -DRUBY_UNTYPED_DATA_WARNING=0'
  # Ruby 3.4 headers intentionally use compatibility declarations that recent
  # Clang diagnoses under -Wpedantic. Keep those from obscuring extension warnings.
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
