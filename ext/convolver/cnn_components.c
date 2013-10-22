// ext/convolver/cnn_components.c

#include <xmmintrin.h>
#include "cnn_components.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Run a single fully-connected layer, calculating output from input
//
//    Benchmark: 1024 inputs, 256 outputs. 1000 iterations. 0.56 seconds
//
//

void nn_run_layer_raw( int in_size, int out_size,
    float *in_ptr, float *weights, float *thresholds, float *out_ptr ) {
  int i, j, in_aligned_size, out_aligned_size, offset;
  __m128 simd_x, simd_y, simd_t;

  in_aligned_size = 4 * ( in_size/4 );
  out_aligned_size = 4 * ( out_size/4 );

  // Calculate activation
  for ( i = 0; i < out_size; i++ ) {

    float t = 0.0;
    simd_t = _mm_setzero_ps();
    offset = i * in_size;

    // Use SIMD for all the aligned values in groups of 4
    for ( j = 0; j < in_aligned_size; j +=4 ) {
      simd_x = _mm_load_ps( in_ptr + j );
      // Weights might not align to 16 bytes due to size of layers
      simd_y = _mm_loadu_ps( weights + (offset + j) );
      simd_x = _mm_mul_ps( simd_x, simd_y );
      simd_t = _mm_add_ps( simd_x, simd_t );
    }

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = in_aligned_size; j < in_size; j++ ) {
      t += in_ptr[ j ] * weights[ offset + j ];
    }

    out_ptr[i] = simd_t[0] + simd_t[1] + simd_t[2] + simd_t[3] + t;
  }

  for ( i = 0; i < out_size; i++ ) {
    out_ptr[i] -= thresholds[i];
    if ( out_ptr[i] < 0.0 ) { out_ptr[i] = 0.0; }
  }

  return;
}
