// ext/convolver/convolve_raw.c

#include "convolve_raw.h"

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
//  Convolve
//
//    Benchmark: 640x480 image, 8x8 kernel, 1000 iterations. 12.3 seconds.
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
    float v[4];
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
    _mm_store_ps( v, simd_t );

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = kernel_aligned; j < kernel_size; j++ ) {
      t += in_ptr[ offset + kernel_co_incr_cache[j] ] * kernel_ptr[ j ];
    }

    out_ptr[i] = v[0] + v[1] + v[2] + v[3] + t;
  }

  xfree( kernel_co_incr_cache );
  return;
}
