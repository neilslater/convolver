// ext/convolver/narray_shared.c

#include "narray_shared.h"

// This is copied from na_array.c, with safety checks and temp vars removed
int na_quick_idxs_to_pos( int rank, int *shape, int *idxs ) {
  int i, pos = 0;
  for ( i = rank - 1; i >= 0; i-- ) {
    pos = pos * shape[i] + idxs[i];
  }
  return pos;
}

// This is inverse of above
void na_quick_pos_to_idxs( int rank, int *shape, int pos, int *idxs ) {
  int i;
  for ( i = 0; i < rank; i++ ) {
    idxs[ i ] = pos % shape[i];
    pos /= shape[i];
  }
  return;
}

// used to place kernel data into array for FFTW3 processing
void fit_backwards_raw( int rank, int *dst_shape, float *dst, int *src_shape, float *src, int *shift_shape ) {
  int i, j, size, x;
  int k_idx[16], dst_idx[16];

  size = 1;
  for ( j = 0; j < rank; j++ ) { size *= src_shape[j]; }

  for ( i = 0; i < size; i++ ) {
    na_quick_pos_to_idxs( rank, src_shape, i, k_idx );
    for ( j = 0; j < rank; j++ ) {
      x =  src_shape[j] - shift_shape[j] - k_idx[j] - 1;
      if ( x < 0 ) x = x + dst_shape[j];
      dst_idx[j] = x;
    }
    dst[ na_quick_idxs_to_pos( rank, dst_shape, dst_idx ) ] = src[i];
  }
  return;
}