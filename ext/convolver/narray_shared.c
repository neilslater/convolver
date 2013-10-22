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
