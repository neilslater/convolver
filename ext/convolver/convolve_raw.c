// ext/convolver/convolve_raw.c

#include "convolve_raw.h"

static inline size_t checked_add( size_t left, size_t right, const char *description ) {
  if ( right > SIZE_MAX - left ) {
    rb_raise( rb_eRangeError, "%s exceeds native implementation limit", description );
  }
  return left + right;
}

static inline size_t checked_multiply( size_t left, size_t right, const char *description ) {
  if ( left != 0 && right > SIZE_MAX / left ) {
    rb_raise( rb_eRangeError, "%s exceeds native implementation limit", description );
  }
  return left * right;
}

static inline size_t size_from_shape( int rank, const size_t *shape, const char *description ) {
  size_t size = 1;
  int i;
  for ( i = 0; i < rank; i++ ) {
    size = checked_multiply( size, shape[i], description );
  }
  return size;
}

// Sets reverse indices
static inline void corner_reset( int rank, const size_t *shape, size_t *rev_indices ) {
  int i;
  for ( i = 0; i < rank; i++ ) { rev_indices[i] = shape[i] - 1; }
}

// Counts indices down, returns number of ranks that reset
static inline int corner_dec( int rank, const size_t *shape, size_t *rev_indices ) {
  int i = 0;
  (void) rank;
  while ( ! rev_indices[i]-- ) {
    rev_indices[i] = shape[i] - 1;
    i++;
  }
  return i;
}

// Generates co-increment steps by rank boundaries crossed, for the outer position as inner position is incremented by 1
static inline void calc_co_increment(
    int rank, const size_t *outer_shape, const size_t *inner_shape, size_t *co_increment ) {
  size_t factor = 1;
  int i;
  co_increment[0] = 1; // co-increment is always 1 in lowest rank
  for ( i = 0; i < rank; i++ ) {
    size_t skipped = checked_multiply( factor, outer_shape[i] - inner_shape[i], "array offset" );
    co_increment[i+1] = checked_add( co_increment[i], skipped, "array offset" );
    factor = checked_multiply( factor, outer_shape[i], "array size" );
  }
}

static inline size_t maximum_offset(
    int rank, const size_t *outer_shape, const size_t *inner_shape, const char *description ) {
  size_t factor = 1;
  size_t offset = 0;
  int i;
  for ( i = 0; i < rank; i++ ) {
    size_t dimension_offset = checked_multiply( factor, inner_shape[i] - 1, description );
    offset = checked_add( offset, dimension_offset, description );
    factor = checked_multiply( factor, outer_shape[i], "array size" );
  }
  return offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Convolve
//
//    Benchmark: 640x480 image, 8x8 kernel, 1000 iterations. 12.3 seconds.
//

void convolve_raw(
    int in_rank, const size_t *in_shape, const float *in_ptr,
    int kernel_rank, const size_t *kernel_shape, const float *kernel_ptr,
    int out_rank, const size_t *out_shape, float *out_ptr ) {
  size_t i, j, input_size, kernel_size, kernel_aligned, out_size, offset;
  size_t maximum_input_offset;
  size_t out_co_incr[LARGEST_RANK + 1], kernel_co_incr[LARGEST_RANK + 1];
  size_t ker_q[LARGEST_RANK], out_q[LARGEST_RANK];
  size_t *kernel_co_incr_cache;
  VALUE cache_storage = 0;

  kernel_size = size_from_shape( kernel_rank, kernel_shape, "kernel size" );
  kernel_aligned = kernel_size - kernel_size % 4;
  out_size = size_from_shape( out_rank, out_shape, "output size" );
  input_size = size_from_shape( in_rank, in_shape, "input size" );

  calc_co_increment( in_rank, in_shape, out_shape, out_co_incr );
  calc_co_increment( in_rank, in_shape, kernel_shape, kernel_co_incr );
  maximum_input_offset = checked_add(
    maximum_offset( in_rank, in_shape, out_shape, "input offset" ),
    maximum_offset( in_rank, in_shape, kernel_shape, "input offset" ),
    "input offset"
  );
  if ( maximum_input_offset >= input_size ) {
    rb_raise( rb_eRangeError, "input offset exceeds native implementation limit" );
  }

  kernel_co_incr_cache = RB_ALLOCV_N( size_t, cache_storage, kernel_size );
  kernel_co_incr_cache[0] = 0;

  corner_reset( kernel_rank, kernel_shape, ker_q );
  for ( i = 1; i < kernel_size; i++ ) {
    kernel_co_incr_cache[i] = checked_add(
      kernel_co_incr_cache[i-1],
      kernel_co_incr[ corner_dec( kernel_rank, kernel_shape, ker_q ) ],
      "kernel offset"
    );
  }

  offset = 0;
  corner_reset( out_rank, out_shape, out_q );

  // Main convolve loop
  for ( i = 0; i < out_size; i++ ) {
#if CONVOLVER_USE_SSE
    __m128 simd_x, simd_y, simd_t;
    float v[4];
    simd_t = _mm_setzero_ps();
#endif
    float t = 0.0;

#if CONVOLVER_USE_SSE
    // Use SIMD for all the aligned values in groups of 4
    for ( j = 0; j < kernel_aligned; j +=4 ) {
      simd_x = _mm_load_ps( kernel_ptr + j );
      // Yes the backwards alignment is correct
      simd_y = _mm_set_ps( in_ptr[ offset + kernel_co_incr_cache[j+3] ], in_ptr[ offset + kernel_co_incr_cache[j+2] ],
                           in_ptr[ offset + kernel_co_incr_cache[j+1] ], in_ptr[ offset + kernel_co_incr_cache[j] ] );
      simd_x = _mm_mul_ps( simd_x, simd_y );
      simd_t = _mm_add_ps( simd_x, simd_t );
    }
    _mm_store_ps( v, simd_t );
#else
    // SSE is unavailable on architectures such as Apple Silicon.
    // Use the same calculation without x86-specific intrinsics.
    for ( j = 0; j < kernel_aligned; j++ ) {
      t += in_ptr[ offset + kernel_co_incr_cache[j] ] * kernel_ptr[j];
    }
#endif

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = kernel_aligned; j < kernel_size; j++ ) {
      t += in_ptr[ offset + kernel_co_incr_cache[j] ] * kernel_ptr[ j ];
    }

#if CONVOLVER_USE_SSE
    out_ptr[i] = v[0] + v[1] + v[2] + v[3] + t;
#else
    out_ptr[i] = t;
#endif

    if ( i + 1 < out_size ) {
      offset = checked_add(
        offset,
        out_co_incr[ corner_dec( out_rank, out_shape, out_q ) ],
        "input offset"
      );
    }
  }

  RB_ALLOCV_END( cache_storage );
}
