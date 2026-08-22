// ext/convolver/correlate_raw.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CORRELATE_RAW_H
#define CORRELATE_RAW_H

#ifndef CONVOLVER_RAW_CONFIG
#define CONVOLVER_RAW_CONFIG
#include <ruby.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#define CONVOLVER_USE_SSE 1
#else
#define CONVOLVER_USE_SSE 0
#endif
#define LARGEST_RANK 16
#endif

void correlate_raw(
    int in_rank, const size_t *in_shape, const float *in_ptr,
    int kernel_rank, const size_t *kernel_shape, const float *kernel_ptr,
    int out_rank, const size_t *out_shape, float *out_ptr );

#endif
