"""Strided layouts: where a value's elements live, and views that move no data.

Every view operation (permute, view, expand, slice, ...) only rewrites a Layout.
A rule returns None when the result cannot be described by strides over the
source's memory; the lowering then copies the source into dense storage and
applies the rule again.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


def row_major(shape) -> tuple[int, ...]:
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= extent
    return tuple(reversed(strides))


@dataclass(frozen=True)
class Layout:
    """Element ``index`` of the value is ``buffer[offset + sum(index[d] * strides[d])]``."""
    buffer: str
    offset: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    @staticmethod
    def of(buffer: str, shape) -> "Layout":
        return Layout(buffer, 0, tuple(shape), row_major(shape))

    @property
    def dense(self) -> bool:
        """Elements are consecutive in row-major order (strides of extent-1 dimensions do not matter)."""
        return all(n == 1 or s == r for n, s, r in zip(self.shape, self.strides, row_major(self.shape)))


def _dim(dim, rank):
    return dim + rank if dim < 0 else dim


def identity(l, shape, **_):
    return l


def permute(l, shape, dims):
    dims = [_dim(d, len(l.shape)) for d in dims]
    return replace(l, shape=tuple(l.shape[d] for d in dims), strides=tuple(l.strides[d] for d in dims))


def transpose(l, shape, dim0=0, dim1=1):
    dims = list(range(len(l.shape)))
    if len(dims) > 1:
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
    return permute(l, shape, dims)


def view(l, shape, size=None):
    """PyTorch's rule: each run of the new shape must cover whole contiguous runs of the old one."""
    old = [(n, s) for n, s in zip(l.shape, l.strides) if n != 1]
    strides, position = [0] * len(shape), len(shape) - 1
    while old:
        # Take the longest contiguous tail of the old dimensions.
        run, step = 1, old[-1][1]
        while old and old[-1][1] == step * run:
            run *= old.pop()[0]
        covered = 1
        while position >= 0 and (covered < run or shape[position] == 1):
            strides[position] = step * covered
            covered *= shape[position]
            position -= 1
        if covered != run:
            return None
    return Layout(l.buffer, l.offset, tuple(shape), tuple(strides))


def expand(l, shape, size=None, **_):
    lead = len(shape) - len(l.shape)
    strides = [0] * lead + [0 if n == 1 and m != 1 else s for n, s, m in zip(l.shape, l.strides, shape[lead:])]
    return replace(l, shape=tuple(shape), strides=tuple(strides))


def squeeze(l, shape, dims=None):
    rank = len(l.shape)
    dims = range(rank) if dims is None else [_dim(d, rank) for d in ([dims] if isinstance(dims, int) else dims)]
    keep = [d for d in range(rank) if not (d in dims and l.shape[d] == 1)]
    return replace(l, shape=tuple(l.shape[d] for d in keep), strides=tuple(l.strides[d] for d in keep))


def unsqueeze(l, shape, dim):
    dim = _dim(dim, len(l.shape) + 1)
    return replace(l, shape=l.shape[:dim] + (1,) + l.shape[dim:], strides=l.strides[:dim] + (0,) + l.strides[dim:])


def slice_(l, shape, dim=0, start=None, end=None, step=1):
    dim, extent = _dim(dim, len(l.shape)), l.shape[_dim(dim, len(l.shape))]
    start = 0 if start is None else min(max(start + extent if start < 0 else start, 0), extent)
    strides = l.strides[:dim] + (l.strides[dim] * step,) + l.strides[dim + 1:]
    return Layout(l.buffer, l.offset + start * l.strides[dim], tuple(shape), strides)


def select(l, shape, dim, index):
    dim = _dim(dim, len(l.shape))
    index = index + l.shape[dim] if index < 0 else index
    return Layout(l.buffer, l.offset + index * l.strides[dim], l.shape[:dim] + l.shape[dim + 1:],
                  l.strides[:dim] + l.strides[dim + 1:])


def as_strided(l, shape, size, stride, storage_offset=None):
    """Strides over the source's storage, which must itself be dense."""
    if not l.dense:
        return None
    return Layout(l.buffer, l.offset + (storage_offset or 0), tuple(size), tuple(stride))


VIEWS = {
    "aten.alias.default": identity, "aten.detach.default": identity, "aten.clone.default": identity,
    "aten.lift_fresh_copy.default": identity, "aten.contiguous.default": identity,
    "aten.permute.default": permute, "aten.transpose.int": transpose, "aten.t.default": transpose,
    "aten.view.default": view, "aten._unsafe_view.default": view, "aten.reshape.default": view,
    "aten.expand.default": expand, "aten.squeeze.default": squeeze, "aten.squeeze.dim": squeeze,
    "aten.squeeze.dims": squeeze, "aten.unsqueeze.default": unsqueeze, "aten.slice.Tensor": slice_,
    "aten.select.int": select, "aten.as_strided.default": as_strided,
}
