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
#define LARGEST_RANK 16

void convolve_raw(
    int in_rank, const size_t *in_shape, const float *in_ptr,
    int kernel_rank, const size_t *kernel_shape, const float *kernel_ptr,
    int out_rank, const size_t *out_shape, float *out_ptr );

#endif
