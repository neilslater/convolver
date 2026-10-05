/* Intentionally included once per dtype; operation/dtype dispatch is outside hot loops. */
void RAW_NAME(
    int in_rank, const size_t *in_shape, const RAW_TYPE *in_ptr,
    int kernel_rank, const size_t *kernel_shape, const RAW_TYPE *kernel_ptr,
    int out_rank, const size_t *out_shape, RAW_TYPE *out_ptr ) {
  size_t i, j, input_size, kernel_size, kernel_aligned, out_size, offset;
  size_t maximum_input_offset;
  size_t out_co_incr[LARGEST_RANK + 1], kernel_co_incr[LARGEST_RANK + 1];
  size_t ker_q[LARGEST_RANK], out_q[LARGEST_RANK];
  size_t *kernel_co_incr_cache;
  VALUE cache_storage = 0;

  kernel_size = size_from_shape( kernel_rank, kernel_shape, "kernel size" );
  kernel_aligned = kernel_size - kernel_size % RAW_LANES;
  out_size = size_from_shape( out_rank, out_shape, "output size" );
  input_size = size_from_shape( in_rank, in_shape, "input size" );

  calc_co_increment( in_rank, in_shape, out_shape, out_co_incr );
  calc_co_increment( in_rank, in_shape, kernel_shape, kernel_co_incr );
  maximum_input_offset = checked_add(
    maximum_offset( in_rank, in_shape, out_shape, "input offset" ),
    maximum_offset( in_rank, in_shape, kernel_shape, "input offset" ),
    "input offset"
  );
  if ( maximum_input_offset >= input_size ) {
    rb_raise( rb_eRangeError, "input offset exceeds native implementation limit" );
  }

  kernel_co_incr_cache = RB_ALLOCV_N( size_t, cache_storage, kernel_size );
  kernel_co_incr_cache[0] = 0;

  corner_reset( kernel_rank, kernel_shape, ker_q );
  for ( i = 1; i < kernel_size; i++ ) {
    kernel_co_incr_cache[i] = checked_add(
      kernel_co_incr_cache[i-1],
      kernel_co_incr[ corner_dec( kernel_rank, kernel_shape, ker_q ) ],
      "kernel offset"
    );
  }

  offset = 0;
  corner_reset( out_rank, out_shape, out_q );

  for ( i = 0; i < out_size; i++ ) {
    RAW_TYPE t = 0;
#if RAW_SIMD
    RAW_VECTOR simd_t = RAW_ZERO();
    RAW_TYPE v[RAW_LANES];
    for ( j = 0; j < kernel_aligned; j += RAW_LANES ) {
      RAW_VECTOR simd_x = RAW_LOAD(kernel_ptr + j);
#if RAW_LANES == 4
      RAW_VECTOR simd_y = _mm_set_ps(
        in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j+3)]],
        in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j+2)]],
        in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j+1)]],
        in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j)]]);
#else
      RAW_VECTOR simd_y = _mm_set_pd(
        in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j+1)]],
        in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j)]]);
#endif
      simd_t = RAW_ADD(RAW_MUL(simd_x, simd_y), simd_t);
    }
    RAW_STORE(v, simd_t);
#else
    for ( j = 0; j < kernel_aligned; j++ ) {
      t += in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j)]] * kernel_ptr[j];
    }
#endif
    for ( j = kernel_aligned; j < kernel_size; j++ ) {
      t += in_ptr[offset + kernel_co_incr_cache[RAW_INDEX(j)]] * kernel_ptr[j];
    }
#if RAW_SIMD
#if RAW_LANES == 4
    out_ptr[i] = v[0] + v[1] + v[2] + v[3] + t;
#else
    out_ptr[i] = v[0] + v[1] + t;
#endif
#else
    out_ptr[i] = t;
#endif

    if ( i + 1 < out_size ) {
      offset = checked_add(
        offset,
        out_co_incr[ corner_dec( out_rank, out_shape, out_q ) ],
        "input offset"
      );
    }
  }

  RB_ALLOCV_END( cache_storage );
}
