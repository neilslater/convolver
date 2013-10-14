// ext/convolver/convolver.c

#include <ruby.h>

// To hold the module object
VALUE Convolver = Qnil;

// Returns magic number 80193 as a test
VALUE method_ext_test(VALUE self) {
  return INT2NUM( 80193 );
}

void Init_convolver() {
  Convolver = rb_define_module( "Convolver" );
  rb_define_singleton_method( Convolver, "ext_test", method_ext_test, 0 );
}
