"""Matrix problems described by operation properties only.

A ``MatrixProblem`` is ``C[b] = A[b] @ B[b]`` with static extents, explicit element strides
(``A[b, m, k] = a_root[a_offset + b*a_batch_stride + m*a_row_stride + k*a_col_stride]``),
operand/result dtypes and a result contract. Nothing here names a model, layer or value role;
``role`` is an optional caller annotation that is recorded but never read by selection.

The three products of one linear map ``Y = X W^T`` (W stored [out, in], row-major) are built by
``linear_products``; they are the forward, input-gradient and weight-gradient problems with the
layouts a captured PyTorch graph produces (``aten.t`` views, no copies).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace

DTYPES = ('bfloat16', 'float32')
OUT_MODES = ('store', 'store_f32', 'accumulate_f32', 'store_raw')   # store_raw: unrounded FP32 accumulator (LATCH-RESEARCH-EDITS)


@dataclass(frozen=True)
class Operand:
    """A rank-2/3 strided view of one storage root (element units)."""
    root: str
    row_stride: int
    col_stride: int
    batch_stride: int = 0
    offset: int = 0

    @property
    def unit_cols(self):
        return self.col_stride == 1

    @property
    def unit_rows(self):
        return self.row_stride == 1


@dataclass(frozen=True)
class MatrixProblem:
    m: int
    n: int
    k: int
    a: Operand
    b: Operand
    out: str = 'C'
    batch: int = 1
    dtype: str = 'bfloat16'          # operand dtype (A and B)
    out_dtype: str = 'bfloat16'      # result dtype
    out_mode: str = 'store'          # store | store_f32 (widened rounded BF16) | accumulate_f32 (C += AB) | store_raw (unrounded FP32)
    role: str = ''                   # recorded annotation only; never consulted by selection

    def __post_init__(self):
        if self.dtype not in DTYPES or self.out_dtype not in DTYPES:
            raise ValueError(f'unsupported dtype {self.dtype}->{self.out_dtype}')
        if self.out_mode not in OUT_MODES:
            raise ValueError(f'unknown result mode {self.out_mode}')
        if min(self.m, self.n, self.k, self.batch) < 1:
            raise ValueError('extents must be positive')

    # ---- derived structural properties used by admission rules -------------------------------------------
    @property
    def ak(self):
        """A is K-contiguous (row-major [M, K])."""
        return self.a.col_stride == 1

    @property
    def bkc(self):
        """B is K-contiguous (B^T row-major, i.e. a [N, K] weight read through a transpose view)."""
        return self.b.row_stride == 1

    @property
    def flops(self):
        return 2 * self.batch * self.m * self.n * self.k

    def a_shape(self):
        return (self.batch, self.m, self.k) if self.batch > 1 else (self.m, self.k)

    def b_shape(self):
        return (self.batch, self.k, self.n) if self.batch > 1 else (self.k, self.n)

    def out_shape(self):
        return (self.batch, self.m, self.n) if self.batch > 1 else (self.m, self.n)

    def a_strides(self):
        s = (self.a.row_stride, self.a.col_stride)
        return (self.a.batch_stride,) + s if self.batch > 1 else s

    def b_strides(self):
        s = (self.b.row_stride, self.b.col_stride)
        return (self.b.batch_stride,) + s if self.batch > 1 else s

    def layouts(self):
        """Frozen-compiler ``TensorLayout`` objects (imported lazily: ops pulls in torch)."""
        from megakernel_lab.operation_compiler.lower import TensorLayout
        return (TensorLayout(self.a_shape(), self.a_strides(), self.a.root, self.a.offset),
                TensorLayout(self.b_shape(), self.b_strides(), self.b.root, self.b.offset))

    def key(self):
        """Hashable structural identity (excludes root names and the role annotation)."""
        strip = lambda o: (o.row_stride, o.col_stride, o.batch_stride, o.offset)
        return (self.m, self.n, self.k, self.batch, strip(self.a), strip(self.b),
                self.dtype, self.out_dtype, self.out_mode)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d['a'], d['b'] = Operand(**d['a']), Operand(**d['b'])
        return cls(**d)

    def renamed(self, a=None, b=None, out=None):
        return replace(self, a=replace(self.a, root=a or self.a.root), b=replace(self.b, root=b or self.b.root),
                       out=out or self.out)


def dense(m, n, k, *, ak=True, bkc=False, batch=1, dtype='bfloat16', out_dtype=None, out_mode='store',
          a='A', b='B', out='C', role='', a_pad=0, b_pad=0):
    """A contiguous (or transposed-view) dense problem; ``*_pad`` widens the leading dimension (gapped rows)."""
    out_dtype = out_dtype or dtype
    lda = (k if ak else m) + a_pad
    ldb = (k if bkc else n) + b_pad
    # A storage is [batch, M, lda] when K-contiguous, else a transposed view of [batch, K, lda];
    # B storage is a transposed view of [batch, N, ldb] when K-contiguous, else [batch, K, ldb].
    oa = Operand(a, lda if ak else 1, 1 if ak else lda, lda * (m if ak else k) if batch > 1 else 0)
    ob = Operand(b, 1 if bkc else ldb, ldb if bkc else 1, ldb * (n if bkc else k) if batch > 1 else 0)
    return MatrixProblem(m, n, k, oa, ob, out=out, batch=batch, dtype=dtype, out_dtype=out_dtype,
                         out_mode=out_mode, role=role)


def linear_products(tokens, d_in, d_out, *, dtype='bfloat16', grad_mode='store'):
    """Forward/dX/dW problems of ``Y[T, out] = X[T, in] @ W[out, in]^T`` as a captured graph lays them out.

    * forward: A = X (K-contiguous), B = W^T view (K-contiguous: row stride 1);
    * dX = dY @ W: A = dY (K-contiguous), B = W (N-contiguous);
    * dW = dY^T @ X: A = dY^T view (M-contiguous), B = X (N-contiguous).
    """
    fwd = dense(tokens, d_out, d_in, ak=True, bkc=True, dtype=dtype, a='X', b='W', out='Y', role='forward')
    dx = dense(tokens, d_in, d_out, ak=True, bkc=False, dtype=dtype, a='dY', b='W', out='dX', role='grad_input')
    dw = dense(d_out, d_in, tokens, ak=False, bkc=False, dtype=dtype, a='dY', b='X', out='dW', role='grad_weight',
               out_mode=grad_mode, out_dtype='float32' if grad_mode != 'store' else dtype)
    return fwd, dx, dw


def expert_products(experts, capacity, d_model, d_ff, *, dtype='bfloat16'):
    """Batched (grouped, equal-capacity) expert FFN up-projection products: forward, dX and dW.

    Expert e computes ``H[e] = X[e] @ W1[e]`` with X [E, C, D] and W1 [E, D, F]; capacity rows per expert are
    padded to C by the caller (expert-choice routing gives exactly C rows).
    """
    fwd = dense(capacity, d_ff, d_model, ak=True, bkc=False, batch=experts, dtype=dtype,
                a='Xe', b='W1', out='H', role='forward')
    dx = dense(capacity, d_model, d_ff, ak=True, bkc=True, batch=experts, dtype=dtype,
               a='dH', b='W1', out='dXe', role='grad_input')
    dw = dense(d_model, d_ff, capacity, ak=False, bkc=False, batch=experts, dtype=dtype,
               a='Xe', b='dH', out='dW1', role='grad_weight')
    return fwd, dx, dw
