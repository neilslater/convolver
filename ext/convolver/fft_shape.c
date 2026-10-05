#include "fft_shape.h"
#include "raw_config.h"
#include <limits.h>
#include <math.h>

static size_t positive_size(VALUE value) {
  if (!RB_INTEGER_TYPE_P(value) || RTEST(rb_funcall(value, rb_intern("<="), 1, INT2FIX(0)))) {
    rb_raise(rb_eArgError, "FFT dimensions and size limit must be positive integers");
  }
  return NUM2SIZET(value);
}

static long read_shape(VALUE shape, size_t *dimensions) {
  Check_Type(shape, T_ARRAY);
  long rank = RARRAY_LEN(shape);
  if (rank < 1 || rank > LARGEST_RANK) {
    rb_raise(rb_eArgError, "FFT shape rank must be between 1 and %d", LARGEST_RANK);
  }
  for (long axis = 0; axis < rank; axis++) {
    dimensions[axis] = positive_size(rb_ary_entry(shape, axis));
  }
  return rank;
}

static size_t doubled_candidate(size_t initial, size_t target, int even, size_t maximum) {
  size_t candidate = initial;
  while (candidate < target || (even && candidate % 2 != 0)) {
    if (candidate > maximum / 2) return 0;
    candidate *= 2;
  }
  return candidate;
}

/* Enumerate 3/5 powers only while they can improve the best 2/3/5 candidate. */
static size_t next_fast_size(size_t target, int even, size_t maximum) {
  size_t best = 0, five = 1;
  for (;;) {
    size_t three = five;
    for (;;) {
      size_t candidate = doubled_candidate(three, target, even, maximum);
      if (candidate != 0 && (best == 0 || candidate < best)) best = candidate;
      size_t bound = best != 0 ? best : maximum;
      if (three > bound / 3) break;
      three *= 3;
    }
    size_t bound = best != 0 ? best : maximum;
    if (five > bound / 5) break;
    five *= 5;
  }
  return best;
}

/*
 * Generates optional smooth dimensions without allocating numeric buffers.
 * @private
 */
static VALUE convolver_fft_fast_shape(VALUE self, VALUE shape, VALUE maximum_value) {
  (void)self;
  size_t dimensions[LARGEST_RANK];
  long rank = read_shape(shape, dimensions);
  size_t maximum = positive_size(maximum_value);
  for (long axis = 0; axis < rank; axis++) {
    if (dimensions[axis] > maximum) return Qnil;
    dimensions[axis] = next_fast_size(dimensions[axis], axis == rank - 1, maximum);
    if (dimensions[axis] == 0) return Qnil;
  }
  VALUE result = rb_ary_new_capa(rank);
  for (long axis = 0; axis < rank; axis++) {
    rb_ary_push(result, SIZET2NUM(dimensions[axis]));
  }
  return result;
}

static double large_factor_penalty(size_t value) {
  double penalty = 0.0;
  size_t divisor = 2;
  while (divisor <= value / divisor) {
    while (value % divisor == 0) {
      if (divisor > 7) penalty += 1.5 * (log2((double)divisor) - 3.0);
      value /= divisor;
    }
    divisor += divisor == 2 ? 1 : 2;
  }
  if (value > 7) penalty += 1.5 * (log2((double)value) - 3.0);
  return penalty;
}

/*
 * Scores an admissible candidate; Ruby owns the complete FFT buffer policy.
 * @private
 */
static VALUE convolver_fft_shape_cost(VALUE self, VALUE shape) {
  (void)self;
  size_t dimensions[LARGEST_RANK], product = 1;
  long rank = read_shape(shape, dimensions);
  for (long axis = 0; axis < rank; axis++) {
    /* Bound trial factorization even if this private primitive is called directly. */
    if (dimensions[axis] > (size_t)INT_MAX / 16) {
      rb_raise(rb_eRangeError, "FFT scoring axis exceeds native integer limit");
    }
    if (dimensions[axis] > SIZE_MAX / product) {
      rb_raise(rb_eRangeError, "FFT scoring size exceeds native element limit");
    }
    product *= dimensions[axis];
  }
  double cost = 0.0;
  for (long axis = 0; axis < rank; axis++) {
    cost += log2((double)dimensions[axis]) + large_factor_penalty(dimensions[axis]);
  }
  return DBL2NUM((double)product * cost);
}

void convolver_init_fft_shape(VALUE convolver) {
  rb_define_singleton_method(convolver, "fft_fast_shape", convolver_fft_fast_shape, 2);
  rb_define_singleton_method(convolver, "fft_shape_cost", convolver_fft_shape_cost, 1);
}
