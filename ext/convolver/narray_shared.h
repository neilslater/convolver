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
int na_quick_idxs_to_pos( int rank, int *shape, int *idxs );

// This is inverse of above
void na_quick_pos_to_idxs( int rank, int *shape, int pos, int *idxs );

void fit_backwards_raw( int rank, int *dst_shape, float *dst, int *src_shape, float *src, int *shift_shape );

#endif
