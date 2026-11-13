"""View rules against PyTorch's own strides, on random shapes."""
import itertools
import random

import torch

from latch.layout import Layout, expand, permute, select, slice_, squeeze, unsqueeze, view


def addresses(layout):
    return [layout.offset + sum(i * s for i, s in zip(index, layout.strides))
            for index in itertools.product(*(range(n) for n in layout.shape))]


def torch_addresses(tensor):
    return [tensor.storage_offset() + sum(i * s for i, s in zip(index, tensor.stride()))
            for index in itertools.product(*(range(n) for n in tensor.shape))]


def random_view(rng):
    """A random strided tensor over a dense buffer, and the Layout that should describe it."""
    shape = tuple(rng.choice([1, 2, 3, 4]) for _ in range(rng.randint(0, 4)))
    tensor, layout = torch.zeros(shape), Layout.of("b", shape)
    for _ in range(rng.randint(0, 3)):
        rank, choice = tensor.dim(), rng.random()
        if choice < 0.35 and rank:
            order = rng.sample(range(rank), rank)
            tensor, layout = tensor.permute(order), permute(layout, None, order)
        elif choice < 0.5 and rank:
            dim = rng.randrange(rank)
            start, step = rng.randrange(tensor.shape[dim]), rng.choice([1, 2])
            tensor = tensor.narrow(dim, start, tensor.shape[dim] - start)[(slice(None),) * dim + (slice(None, None, step),)]
            layout = slice_(layout, tuple(tensor.shape), dim, start, None, step)
        elif choice < 0.65 and rank:
            dim = rng.randrange(rank)
            index = rng.randrange(tensor.shape[dim])
            tensor, layout = tensor.select(dim, index), select(layout, None, dim, index)
        elif choice < 0.8:
            dim = rng.randint(0, rank)
            tensor, layout = tensor.unsqueeze(dim), unsqueeze(layout, None, dim)
        elif choice < 0.9:
            size = tuple(rng.choice([2, 3]) if n == 1 else n for n in tensor.shape)
            tensor, layout = tensor.expand(size), expand(layout, size)
        else:
            tensor, layout = tensor.squeeze(), squeeze(layout, None)
    return tensor, layout


def test_views_address_the_same_elements_as_torch():
    rng = random.Random(0)
    for _ in range(3000):
        tensor, layout = random_view(rng)
        assert layout.shape == tuple(tensor.shape)
        assert addresses(layout) == torch_addresses(tensor)
        assert layout.dense == tensor.is_contiguous()


def test_view_agrees_with_torch_on_what_is_viewable():
    rng = random.Random(1)
    for _ in range(3000):
        tensor, layout = random_view(rng)
        factors, rest = [], tensor.numel()
        while rest > 1:
            factor = rng.choice([d for d in (2, 3, 4, 6, 8, 9, 12, rest) if rest % d == 0])
            factors.append(factor)
            rest //= factor
        for _ in range(rng.randint(0, 2)):
            factors.insert(rng.randint(0, len(factors)), 1)
        try:
            expected = tensor.view(tuple(factors))
        except RuntimeError:
            expected = None
        actual = view(layout, tuple(factors))
        assert (actual is None) == (expected is None)
        if actual is not None:
            assert addresses(actual) == torch_addresses(expected)
