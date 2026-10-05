#ifndef CORRELATE_RAW_H
#define CORRELATE_RAW_H
#include "raw_config.h"

void correlate_raw(
    int in_rank, const size_t *in_shape, const float *in_ptr,
    int kernel_rank, const size_t *kernel_shape, const float *kernel_ptr,
    int out_rank, const size_t *out_shape, float *out_ptr );

void correlate_raw_double(
    int in_rank, const size_t *in_shape, const double *in_ptr,
    int kernel_rank, const size_t *kernel_shape, const double *kernel_ptr,
    int out_rank, const size_t *out_shape, double *out_ptr );
#endif
