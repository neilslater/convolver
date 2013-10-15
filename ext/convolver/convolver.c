// ext/convolver/convolver.c

#include <ruby.h>
#include "narray.h"

// This is copied from na_array.c, with safety checks and temp vars removed
inline int na_quick_idxs_to_pos( int rank, int *shape, int *idxs ) {
  int i, pos = 0;
  for ( i = rank; (i--)>0; ) {
    pos = pos * shape[i] + idxs[i];
  }
  return pos;
}

// This is inverse of above
inline void na_quick_pos_to_idxs( int rank, int *shape, int pos, int *idxs ) {
  int i;
  for ( i = 0; i < rank; i++ ) {
    idxs[ i ] = pos % shape[i];
    pos /= shape[i];
  }
  return;
}

/* Ruby versions of above, to test logic
def na_quick_idxs_to_pos( rank, shape, idxs )
  pos = 0
  (rank-1).downto(0) do |i|
    pos = pos * shape[i] + idxs[i];
  end
  pos
end

def na_quick_pos_to_idxs( rank, shape, pos, idxs )
  remainder = pos
  (0...rank).each do |i|
    idxs[ i ] = remainder % shape[i]
    remainder /= shape[i]
  end
  idxs
end

shape = [5,7,11]
idxs = []
(0...5*7*11).all? { |p| na_quick_pos_to_idxs( 3, shape, p, idxs ); na_quick_idxs_to_pos( 3, shape, idxs ) == p }
 => true
*/


// To hold the module object
VALUE Convolver = Qnil;

static VALUE narray_convolve( VALUE self, VALUE a, VALUE b ) {
  struct NARRAY *na_a, *na_b, *na_c;
  volatile VALUE val_a, val_b, val_c;
  int target_rank, i, j, k, offset;
  int target_shape[16];
  float *in_a, *in_b, *out;
  register float t;
  int out_idx[16];
  int in_a_idx[16];
  int in_b_idx[16];

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

  // Get quick pointers
  in_a = (float*) na_a->ptr;
  in_b = (float*) na_b->ptr;
  out = (float*) na_c->ptr;

  // Calculate output in pointer sequence order, derive dimension data from that.
  for ( i = 0; i < na_c->total; i++ ) {
    t = 0.0;
    na_quick_pos_to_idxs( target_rank, target_shape, i, out_idx );
    for ( j = 0; j < na_b->total; j++ ) {
      na_quick_pos_to_idxs( target_rank, na_b->shape, j, in_b_idx );
      for ( k = 0; k < target_rank; k++ ) { in_a_idx[k] = out_idx[k] + in_b_idx[k]; }
      offset = na_quick_idxs_to_pos( target_rank, na_a->shape, in_a_idx );
      t += in_a[ offset ] * in_b[ j ];
    }
    out[i] = t;
  }

  return val_c;
}

void Init_convolver() {
  Convolver = rb_define_module( "Convolver" );
  rb_define_singleton_method( Convolver, "convolve", narray_convolve, 2 );
}
