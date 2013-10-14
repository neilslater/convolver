// ext/convolver/convolver.c

#include <ruby.h>
#include "narray.h"

// To hold the module object
VALUE Convolver = Qnil;

static VALUE narray_convolve( VALUE self, VALUE a, VALUE b ) {
  struct NARRAY *na_a, *na_b, *na_c;
  volatile VALUE val_a, val_b; val_c;
  int target_rank, i;
  int target_shape[16];

  val_a = na_cast_object(a, NA_SFLOAT);
  GetNArray( val_a, na_a );

  val_b = na_cast_object(b, NA_SFLOAT);
  GetNArray( val_b, na_b );

  if ( na_a->rank < na_b->rank ) {
    rb_raise( rb_eArgError, "narray b must have equal or lower rank than narray a" );
  }

  if ( na_a->rank < na_b->rank ) {
    rb_raise( rb_eArgError, "narray a must have equal rank to narray b (temporary restriction)" );
  }

  if ( na_a->rank > 16 ) {
    rb_raise( rb_eArgError, "exceeded maximum narray rank for convolve of 16" );
  }

  target_rank = na_a->rank;

  for ( i = 0; i < target_rank; i++ ) {
    target_shape[i] = na_a->shape[i] - na_b->shape[i] + 1;
    if ( target_shape[i] < 1 ) {
      rb_raise( rb_eArgError, "narray b is bigger in one or more dimensions than narray a" );
    }
  }

  val_c = na_make_object( NA_SFLOAT, target_rank, target_shape, CLASS_OF( val_a ) );
  GetNArray( val_c, na_c );

  return val_c;
}

void Init_convolver() {
  Convolver = rb_define_module( "Convolver" );
  rb_define_singleton_method( Convolver, "convolve", narray_convolve, 2 );
}
