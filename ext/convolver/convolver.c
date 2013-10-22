// ext/convolver/convolver.c

#include <ruby.h>
#include "narray.h"
#include <stdio.h>
#include <xmmintrin.h>

#include "narray_shared.h"
#include "convolve_raw.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Neural net
//
//    Benchmark: 1024 inputs, 256 outputs. 1000 iterations. 0.56 seconds
//
//

void nn_run_layer_raw( int in_size, int out_size,
    float *in_ptr, float *weights, float *thresholds, float *out_ptr ) {
  int i, j, in_aligned_size, out_aligned_size, offset;
  __m128 simd_x, simd_y, simd_t;

  in_aligned_size = 4 * ( in_size/4 );
  out_aligned_size = 4 * ( out_size/4 );

  // Calculate activation
  for ( i = 0; i < out_size; i++ ) {

    float t = 0.0;
    simd_t = _mm_setzero_ps();
    offset = i * in_size;

    // Use SIMD for all the aligned values in groups of 4
    for ( j = 0; j < in_aligned_size; j +=4 ) {
      simd_x = _mm_load_ps( in_ptr + j );
      // Weights might not align to 16 bytes due to size of layers
      simd_y = _mm_loadu_ps( weights + (offset + j) );
      simd_x = _mm_mul_ps( simd_x, simd_y );
      simd_t = _mm_add_ps( simd_x, simd_t );
    }

    // Complete any remaining 1,2 or 3 items one at a time
    for ( j = in_aligned_size; j < in_size; j++ ) {
      t += in_ptr[ j ] * weights[ offset + j ];
    }

    out_ptr[i] = simd_t[0] + simd_t[1] + simd_t[2] + simd_t[3] + t;
  }

  for ( i = 0; i < out_size; i++ ) {
    out_ptr[i] -= thresholds[i];
    if ( out_ptr[i] < 0.0 ) { out_ptr[i] = 0.0; }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// To hold the module object
VALUE Convolver = Qnil;

/* @overload convolve( signal, kernel )
 * Calculates convolution of an array of floats representing a signal, with a second array representing
 * a kernel. The two parameters must have the same rank. The output has same rank, its size in each dimension d is given by
 *  signal.shape[d] - kernel.shape[d] + 1
 * @param [NArray] signal must be same size or larger than kernel in each dimension
 * @param [NArray] kernel must be same size or smaller than signal in each dimension
 * @return [NArray] result of convolving signal with kernel
 */
static VALUE narray_convolve( VALUE self, VALUE a, VALUE b ) {
  struct NARRAY *na_a, *na_b, *na_c;
  volatile VALUE val_a, val_b, val_c;
  int target_rank, i;
  int target_shape[LARGEST_RANK];

  val_a = na_cast_object(a, NA_SFLOAT);
  GetNArray( val_a, na_a );

  val_b = na_cast_object(b, NA_SFLOAT);
  GetNArray( val_b, na_b );

  if ( na_a->rank != na_b->rank ) {
    rb_raise( rb_eArgError, "narray a must have equal rank to narray b (a rack %d, b rank %d)", na_a->rank,  na_b->rank );
  }

  if ( na_a->rank > LARGEST_RANK ) {
    rb_raise( rb_eArgError, "exceeded maximum narray rank for convolve of %d", LARGEST_RANK );
  }

  target_rank = na_a->rank;

  for ( i = 0; i < target_rank; i++ ) {
    target_shape[i] = na_a->shape[i] - na_b->shape[i] + 1;
    if ( target_shape[i] < 1 ) {
      rb_raise( rb_eArgError, "narray b is bigger in one or more dimensions than narray a" );
    }
  }

  val_c = na_make_object( NA_SFLOAT, target_rank, target_shape, CLASS_OF( val_a ) );
  GetNArray( val_c, na_c );

  convolve_raw(
    target_rank, na_a->shape, (float*) na_a->ptr,
    target_rank, na_b->shape, (float*) na_b->ptr,
    target_rank, target_shape, (float*) na_c->ptr );

  return val_c;
}

/* @overload nn_run_layer( inputs, weights, thresholds )
 * Calculates activations of a fully-connected neural network layer. The transfer function after
 * summing weights and applying threshold is a "ReLU", equivalent to
 *  y = x < 0.0 ? 0.0 : x
 * this is less sophisticated than many neural net architectures, but is fast to calculate and to
 * train.
 * @param [NArray] inputs must be rank 1 array of floats
 * @param [NArray] weights must be rank 2 array of floats, with first dimension size of inputs, and second dimension size equal to number of outputs
 * @param [NArray] thresholds must be rank 1 array of floats, size equal to number of outputs desired
 * @return [NArray] neuron activations
 */
static VALUE narray_nn_run_single_layer( VALUE self, VALUE inputs, VALUE weights, VALUE thresholds ) {
  struct NARRAY *na_inputs, *na_weights, *na_thresholds, *na_outputs;
  volatile VALUE val_inputs, val_weights, val_thresholds, val_outputs;
  int input_size, output_size;
  int output_shape[1];

  val_inputs = na_cast_object(inputs, NA_SFLOAT);
  GetNArray( val_inputs, na_inputs );
  if ( na_inputs->rank != 1 ) {
    rb_raise( rb_eArgError, "input must be array of rank 1" );
  }
  input_size = na_inputs->total;

  val_weights = na_cast_object(weights, NA_SFLOAT);
  GetNArray( val_weights, na_weights );
  if ( na_weights->rank != 2 ) {
    rb_raise( rb_eArgError, "weights must be array of rank 2" );
  }
  if ( na_weights->shape[0] != input_size ) {
    rb_raise( rb_eArgError, "weights shape mismatch, expected %d across, got %d", input_size, na_weights->shape[0] );
  }
  output_size = na_weights->shape[1];

  val_thresholds = na_cast_object(thresholds, NA_SFLOAT);
  GetNArray( val_thresholds, na_thresholds );
  if ( na_thresholds->rank != 1 ) {
    rb_raise( rb_eArgError, "thresholds must be array of rank 1" );
  }
  if ( na_thresholds->shape[0] != output_size ) {
    rb_raise( rb_eArgError, "thresholds expected size %d, but got %d", output_size, na_thresholds->shape[0] );
  }

  output_shape[0] = output_size;
  val_outputs = na_make_object( NA_SFLOAT, 1, output_shape, CLASS_OF( val_inputs ) );
  GetNArray( val_outputs, na_outputs );

  nn_run_layer_raw( input_size, output_size, (float*) na_inputs->ptr, (float*) na_weights->ptr,
      (float*) na_thresholds->ptr, (float*) na_outputs->ptr );

  return val_outputs;
}


void Init_convolver() {
  Convolver = rb_define_module( "Convolver" );
  rb_define_singleton_method( Convolver, "convolve", narray_convolve, 2 );
  rb_define_singleton_method( Convolver, "nn_run_layer", narray_nn_run_single_layer, 3 );
}
