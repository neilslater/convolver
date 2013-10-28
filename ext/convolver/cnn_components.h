// ext/convolver/cnn_components.h

////////////////////////////////////////////////////////////////////////////////////////////////
//
// Declarations of narray helper functions
//

#ifndef CNN_COMPONENTS_H
#define CNN_COMPONENTS_H

void nn_run_layer_raw( int in_size, int out_size,
    float *in_ptr, float *weights, float *thresholds, float *out_ptr );

void max_pool_raw( int rank, int *input_shape, float *input_ptr,
    int *output_shape, float *output_ptr, int tile_by, int pool_by );

#endif
