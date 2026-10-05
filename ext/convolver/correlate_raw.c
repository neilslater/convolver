#include "correlate_raw.h"
#include "raw_helpers.h"

#define RAW_INDEX(j) (j)
#if defined(__SSE__)
#define RAW_SIMD 1
#else
#define RAW_SIMD 0
#endif
#define RAW_NAME correlate_raw
#define RAW_TYPE float
#define RAW_LANES 4
#define RAW_VECTOR __m128
#define RAW_ZERO _mm_setzero_ps
#define RAW_LOAD _mm_loadu_ps
#define RAW_STORE _mm_storeu_ps
#define RAW_ADD _mm_add_ps
#define RAW_MUL _mm_mul_ps
#include "raw_loop.h"
#undef RAW_NAME
#undef RAW_TYPE
#undef RAW_LANES
#undef RAW_VECTOR
#undef RAW_ZERO
#undef RAW_LOAD
#undef RAW_STORE
#undef RAW_ADD
#undef RAW_MUL
#undef RAW_SIMD

#if defined(__SSE2__)
#define RAW_SIMD 1
#else
#define RAW_SIMD 0
#endif
#define RAW_NAME correlate_raw_double
#define RAW_TYPE double
#define RAW_LANES 2
#define RAW_VECTOR __m128d
#define RAW_ZERO _mm_setzero_pd
#define RAW_LOAD _mm_loadu_pd
#define RAW_STORE _mm_storeu_pd
#define RAW_ADD _mm_add_pd
#define RAW_MUL _mm_mul_pd
#include "raw_loop.h"
#undef RAW_NAME
#undef RAW_TYPE
#undef RAW_LANES
#undef RAW_VECTOR
#undef RAW_ZERO
#undef RAW_LOAD
#undef RAW_STORE
#undef RAW_ADD
#undef RAW_MUL
#undef RAW_SIMD

#undef RAW_INDEX
