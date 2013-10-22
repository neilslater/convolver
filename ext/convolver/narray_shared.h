// ext/convolver/narray_shared.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CONVOLVER_NARRAY_SHARED_H
#define CONVOLVER_NARRAY_SHARED_H

#include <ruby.h>
#include "narray.h"

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

#endif
