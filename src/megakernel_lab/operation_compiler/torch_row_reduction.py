"""Opt-in pinned CUDA vectorized FP32 sum/mean tree, without a runtime dependency.

Algorithm reference: PyTorch2.9.1 Reduce.cuh input_vectorized_thread_reduce_impl
and block_x_reduce; ReduceSumProdKernel.cu sum_functor and SharedReduceOps.h
MeanOps. SUM has identity projection; mean alone multiplies by its factor.
Independently expressed here.
Admission proves a contiguous single-axis reduction, four-value vectors, one
32-lane warp per output, and no inter-warp/global reduction for this schedule.
Legacy compiler arithmetic remains the default.
"""
CONTRACT = 'torch_cuda_vector4_v1'


def eligible(*, metadata, target, dtype, source_dtype, source_shape, source_strides,
             source_offset, dims, rows, extent):
    # CUDA vector4 loading peels an unaligned prefix, changing the summation
    # tree. Only aligned contiguous row storage has the tree expressed here.
    contiguous, stride = True, 1
    for size, actual in zip(reversed(source_shape), reversed(source_strides)):
        contiguous &= size == 1 or actual == stride
        stride *= size
    return (metadata.get('row_reduction_contract') == CONTRACT
            and target in {'aten.mean.dim', 'aten.sum.dim_IntList'}
            and dtype == source_dtype == 'float32'
            and len(source_shape) == len(source_strides) and contiguous
            and source_offset % 4 == 0 and stride < 2**31
            and len(dims) == 1 and dims[0] == len(source_shape) - 1
            and rows >= 64 and 128 < extent <= 4096 and extent % 128 == 0)


def row_code(*, out, rows, extent, leaf, rounded, scale):
    if not (rows >= 64 and 128 < extent <= 4096 and extent % 128 == 0):
        raise ValueError('CUDA vector4 row contract admission failed')
    return f'''{{ // explicit arithmetic contract: {CONTRACT}
  const unsigned latch_lo = /*ROW_LO*/0u;
  if (blockIdx.x >= latch_lo) {{
    const unsigned lane = threadIdx.x & 31u, latch_slot = threadIdx.x >> 5;
    constexpr unsigned latch_w0 = 0u, latch_nw = 8u;
    const unsigned latch_nb = gridDim.x - latch_lo;
    for (unsigned latch_base = (blockIdx.x-latch_lo)*8u;
         latch_base < {rows}u; latch_base += latch_nb*8u) {{
      const unsigned g = latch_base + latch_slot;
      if (g < {rows}u) {{
        const unsigned prow = g;
        float latch_acc4[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (unsigned j = 0; j < {extent//128}u; ++j) {{
#pragma unroll
          for (unsigned c = 0; c < 4u; ++c) {{
            const unsigned pcol = 128u*j + 4u*lane + c;
            latch_acc4[c] = latch_acc4[c] + {leaf};
          }}
        }}
        float second = ((latch_acc4[0]+latch_acc4[1])+latch_acc4[2])+latch_acc4[3];
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
          second = second + __shfl_down_sync(0xffffffffu, second, offset);
        second = __shfl_sync(0xffffffffu, second, 0);
        if (lane == 0u) {out}[g] = {rounded('second' + scale)};
/*ROW_MEMBERS*/
      }}
    }}
  }}
}}'''
