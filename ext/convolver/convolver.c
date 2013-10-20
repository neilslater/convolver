// ext/convolver/convolver.c

#include <ruby.h>
#include "narray.h"
#include <stdio.h>
#include <xmmintrin.h>

#define LARGEST_RANK 16

// This is copied from na_array.c, with safety checks and temp vars removed
inline int na_quick_idxs_to_pos( int rank, int *shape, int *idxs ) {
  int i, pos = 0;
  for ( i = rank - 1; i >= 0; i-- ) {
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

inline int size_from_shape( int rank, int *shape ) {
  int size = 1;
  int i;
  for ( i = 0; i < rank; i++ ) { size *= shape[i]; }
  return size;
}

// Sets reverse indices
inline void corner_reset( int rank, int *shape, int *rev_indices ) {
  int i;
  for ( i = 0; i < rank; i++ ) { rev_indices[i] = shape[i] - 1; }
  return;
}

// Counts indices down, returns number of ranks that reset
inline int corner_dec( int rank, int *shape, int *rev_indices ) {
  int i = 0;
  while ( ! rev_indices[i]-- ) {
    rev_indices[i] = shape[i] - 1;
    i++;
  }
  return i;
}

// Generates co-increment steps by rank boundaries crossed, for the outer position as inner position is incremented by 1
inline void calc_co_increment( int rank, int *outer_shape, int *inner_shape, int *co_increment ) {
  int i, factor;
  co_increment[0] = 1; // co-increment is always 1 in lowest rank
  factor = 1;
  for ( i = 0; i < rank; i++ ) {
    co_increment[i+1] = co_increment[i] + factor * ( outer_shape[i] - inner_shape[i] );
    factor *= outer_shape[i];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Convolve method 4.
//
//    Benchmark: 640x480 image, 8x8 kernel, 1000 iterations. 11.54 seconds. Score: 63
//

void convolve_raw(
    int in_rank, int *in_shape, float *in_ptr,
    int kernel_rank, int *kernel_shape, float *kernel_ptr,
    int out_rank, int *out_shape, float *out_ptr ) {
  int i, j, in_size, kernel_size, kernel_aligned, out_size, offset;
  int out_co_incr[LARGEST_RANK], kernel_co_incr[LARGEST_RANK];
  int ker_q[LARGEST_RANK], out_q[LARGEST_RANK];
  int *kernel_co_incr_cache;

  in_size = size_from_shape( in_rank, in_shape );
  kernel_size = size_from_shape( kernel_rank, kernel_shape );
  kernel_aligned = 4 * (kernel_size/4);
  out_size = size_from_shape( out_rank, out_shape );

  calc_co_increment( in_rank, in_shape, out_shape, out_co_incr );
  calc_co_increment( in_rank, in_shape, kernel_shape, kernel_co_incr );

  kernel_co_incr_cache = ALLOC_N( int, kernel_size );
  kernel_co_incr_cache[0] = 0;

  corner_reset( kernel_rank, kernel_shape, ker_q );
  for ( i = 1; i < kernel_size; i++ ) {
    kernel_co_incr_cache[i] = kernel_co_incr_cache[i-1] + kernel_co_incr[ corner_dec( kernel_rank, kernel_shape, ker_q  ) ];
  }

  // For convenience of flow, we set offset to -1 and adjust countdown 1 higher to compensate
  offset = -1;
  corner_reset( out_rank, out_shape, out_q );
  out_q[0]++;

  // Main convolve loop
  for ( i = 0; i < out_size; i++ ) {
    __m128 simd_x, simd_y, simd_t;
    float t = 0.0;
    simd_t = _mm_setzero_ps();

    offset += out_co_incr[ corner_dec( out_rank, out_shape, out_q ) ];

    // Use SIMD for all the aligned values in groups of 4
    for ( j = 0; j < kernel_aligned; j +=4 ) {
      simd_x = _mm_load_ps( kernel_ptr + j );
      // Yes the backwards alignment is correct
      simd_y = _mm_set_ps( in_ptr[ offset + kernel_co_incr_cache[j+3] ], in_ptr[ offset + kernel_co_incr_cache[j+2] ],
                           in_ptr[ offset + kernel_co_incr_cache[j+1] ], in_ptr[ offset + kernel_co_incr_cache[j] ] );
      simd_x = _mm_mul_ps( simd_x, simd_y );
      simd_t = _mm_add_ps( simd_x, simd_t );
    }

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = kernel_aligned; j < kernel_size; j++ ) {
      t += in_ptr[ offset + kernel_co_incr_cache[j] ] * kernel_ptr[ j ];
    }

    out_ptr[i] = simd_t[0] + simd_t[1] + simd_t[2] + simd_t[3] + t;
  }

  xfree( kernel_co_incr_cache );
  return;
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// To hold the module object
VALUE Convolver = Qnil;

static VALUE narray_convolve( VALUE self, VALUE a, VALUE b ) {
  struct NARRAY *na_a, *na_b, *na_c;
  volatile VALUE val_a, val_b, val_c;
  int target_rank, i;
  int target_shape[LARGEST_RANK];

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

  if ( na_a->rank > LARGEST_RANK ) {
    rb_raise( rb_eArgError, "exceeded maximum narray rank for convolve of %d", LARGEST_RANK );
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

  convolve_raw(
    target_rank, na_a->shape, (float*) na_a->ptr,
    target_rank, na_b->shape, (float*) na_b->ptr,
    target_rank, target_shape, (float*) na_c->ptr );

  return val_c;
}

void Init_convolver() {
  Convolver = rb_define_module( "Convolver" );
  rb_define_singleton_method( Convolver, "convolve", narray_convolve, 2 );
}
