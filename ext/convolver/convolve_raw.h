// ext/convolver/convolve_raw.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CONVOLVE_RAW_H
#define CONVOLVE_RAW_H

#include <ruby.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#define CONVOLVER_USE_SSE 1
#else
#define CONVOLVER_USE_SSE 0
#endif
#include "narray_shared.h"

#define LARGEST_RANK 16

void convolve_raw(
    int in_rank, int *in_shape, float *in_ptr,
    int kernel_rank, int *kernel_shape, float *kernel_ptr,
    int out_rank, int *out_shape, float *out_ptr );

#endif
