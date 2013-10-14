// ext/convolver/convolver.c

#include <ruby.h>
#include "narray.h"

// To hold the module object
VALUE Convolver = Qnil;

static VALUE narray_convolve( VALUE self, VALUE a, VALUE b ) {
  return Qnil;
}

void Init_convolver() {
  Convolver = rb_define_module( "Convolver" );
  rb_define_singleton_method( Convolver, "convolve", narray_convolve, 2 );
}
