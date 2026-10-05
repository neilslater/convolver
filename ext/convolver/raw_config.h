#ifndef CONVOLVER_RAW_CONFIG_H
#define CONVOLVER_RAW_CONFIG_H
#include <ruby.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#define LARGEST_RANK 16
#endif
