#ifndef CONVOLVER_RAW_HELPERS_H
#define CONVOLVER_RAW_HELPERS_H

static inline size_t checked_add( size_t left, size_t right, const char *description ) {
  if ( right > SIZE_MAX - left ) {
    rb_raise( rb_eRangeError, "%s exceeds native implementation limit", description );
  }
  return left + right;
}

static inline size_t checked_multiply( size_t left, size_t right, const char *description ) {
  if ( left != 0 && right > SIZE_MAX / left ) {
    rb_raise( rb_eRangeError, "%s exceeds native implementation limit", description );
  }
  return left * right;
}

static inline size_t size_from_shape( int rank, const size_t *shape, const char *description ) {
  size_t size = 1;
  int i;
  for ( i = 0; i < rank; i++ ) {
    size = checked_multiply( size, shape[i], description );
  }
  return size;
}

// Sets reverse indices
static inline void corner_reset( int rank, const size_t *shape, size_t *rev_indices ) {
  int i;
  for ( i = 0; i < rank; i++ ) { rev_indices[i] = shape[i] - 1; }
}

// Counts indices down, returns number of ranks that reset
static inline int corner_dec( int rank, const size_t *shape, size_t *rev_indices ) {
  int i = 0;
  (void) rank;
  while ( ! rev_indices[i]-- ) {
    rev_indices[i] = shape[i] - 1;
    i++;
  }
  return i;
}

// Generates co-increment steps by rank boundaries crossed, for the outer position as inner position is incremented by 1
static inline void calc_co_increment(
    int rank, const size_t *outer_shape, const size_t *inner_shape, size_t *co_increment ) {
  size_t factor = 1;
  int i;
  co_increment[0] = 1; // co-increment is always 1 in lowest rank
  for ( i = 0; i < rank; i++ ) {
    size_t skipped = checked_multiply( factor, outer_shape[i] - inner_shape[i], "array offset" );
    co_increment[i+1] = checked_add( co_increment[i], skipped, "array offset" );
    factor = checked_multiply( factor, outer_shape[i], "array size" );
  }
}

static inline size_t maximum_offset(
    int rank, const size_t *outer_shape, const size_t *inner_shape, const char *description ) {
  size_t factor = 1;
  size_t offset = 0;
  int i;
  for ( i = 0; i < rank; i++ ) {
    size_t dimension_offset = checked_multiply( factor, inner_shape[i] - 1, description );
    offset = checked_add( offset, dimension_offset, description );
    factor = checked_multiply( factor, outer_shape[i], "array size" );
  }
  return offset;
}


#endif
