#include <ruby.h>
#include <numo/narray.h>
#include <numo/intern.h>

#include "correlate_raw.h"
#include "convolve_raw.h"

static VALUE mConvolver;

enum convolver_operation {
  CONVOLVER_CORRELATION,
  CONVOLVER_CONVOLUTION
};

static void copy_shape(int rank, const size_t *source, size_t *target) {
  int i;

  for (i = 0; i < rank; i++) {
    target[i] = source[rank - i - 1];
  }
}

static VALUE convolver_basic_valid(VALUE signal, VALUE kernel, enum convolver_operation operation) {
  volatile VALUE signal_value;
  volatile VALUE kernel_value;
  volatile VALUE result_value;
  narray_t *signal_narray;
  narray_t *kernel_narray;
  int rank;
  int i;
  size_t signal_shape[LARGEST_RANK];
  size_t kernel_shape[LARGEST_RANK];
  size_t result_shape[LARGEST_RANK];
  size_t numo_result_shape[LARGEST_RANK];

  if (!rb_obj_is_kind_of(signal, numo_cNArray) || !rb_obj_is_kind_of(kernel, numo_cNArray)) {
    rb_raise(rb_eArgError, "signal and kernel must be Numo::NArray values");
  }

  signal_value = rb_funcall(numo_cSFloat, rb_intern("cast"), 1, signal);
  kernel_value = rb_funcall(numo_cSFloat, rb_intern("cast"), 1, kernel);
  if (!RTEST(na_check_contiguous(signal_value))) {
    signal_value = na_copy(signal_value);
  }
  if (!RTEST(na_check_contiguous(kernel_value))) {
    kernel_value = na_copy(kernel_value);
  }
  GetNArray(signal_value, signal_narray);
  GetNArray(kernel_value, kernel_narray);

  if (signal_narray->size == 0 || kernel_narray->size == 0) {
    rb_raise(rb_eArgError, "signal and kernel must not be empty");
  }

  if (signal_narray->ndim != kernel_narray->ndim) {
    rb_raise(rb_eArgError, "signal and kernel must have equal rank");
  }
  if (signal_narray->ndim > LARGEST_RANK) {
    rb_raise(rb_eArgError, "maximum supported rank is %d", LARGEST_RANK);
  }

  rank = signal_narray->ndim;
  copy_shape(rank, signal_narray->shape, signal_shape);
  copy_shape(rank, kernel_narray->shape, kernel_shape);

  for (i = 0; i < rank; i++) {
    if (signal_shape[i] < kernel_shape[i]) {
      rb_raise(rb_eArgError, "kernel must not be larger than signal in any dimension");
    }
    result_shape[i] = signal_shape[i] - kernel_shape[i] + 1;
    numo_result_shape[rank - i - 1] = (size_t)result_shape[i];
  }

  result_value = nary_new(numo_cSFloat, rank, numo_result_shape);
  if (operation == CONVOLVER_CORRELATION) {
    correlate_raw(
      rank, signal_shape, (float *)na_get_pointer_for_read(signal_value),
      rank, kernel_shape, (float *)na_get_pointer_for_read(kernel_value),
      rank, result_shape, (float *)na_get_pointer_for_write(result_value)
    );
  } else {
    convolve_raw(
      rank, signal_shape, (float *)na_get_pointer_for_read(signal_value),
      rank, kernel_shape, (float *)na_get_pointer_for_read(kernel_value),
      rank, result_shape, (float *)na_get_pointer_for_write(result_value)
    );
  }

  return result_value;
}

/*
 * Calculates a valid cross-correlation using the direct native implementation.
 *
 * @overload correlate_basic_valid(signal, kernel)
 *   @param signal [Numo::NArray] input values
 *   @param kernel [Numo::NArray] correlation kernel
 *   @return [Numo::SFloat] valid cross-correlation result
 */
static VALUE convolver_correlate_basic_valid(VALUE self, VALUE signal, VALUE kernel) {
  (void)self;
  return convolver_basic_valid(signal, kernel, CONVOLVER_CORRELATION);
}

/*
 * Calculates a valid mathematical convolution using the direct native implementation.
 *
 * @overload convolve_basic_valid(signal, kernel)
 *   @param signal [Numo::NArray] input values
 *   @param kernel [Numo::NArray] convolution kernel
 *   @return [Numo::SFloat] valid mathematical convolution result
 */
static VALUE convolver_convolve_basic_valid(VALUE self, VALUE signal, VALUE kernel) {
  (void)self;
  return convolver_basic_valid(signal, kernel, CONVOLVER_CONVOLUTION);
}

void Init_convolver(void) {
  mConvolver = rb_define_module("Convolver");
  rb_define_singleton_method(mConvolver, "correlate_basic_valid", convolver_correlate_basic_valid, 2);
  rb_define_singleton_method(mConvolver, "convolve_basic_valid", convolver_convolve_basic_valid, 2);
}
