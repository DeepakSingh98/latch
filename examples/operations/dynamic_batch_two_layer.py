"""Ordinary two-layer module with bounded dynamic-batch complete training."""
from __future__ import annotations

import torch

from dynamic_batch_example import main


class TwoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(8, 16)
        self.head = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.head(torch.tanh(self.hidden(x)))


def loss_fn(prediction, target):
    return (prediction - target).square().mean()


if __name__ == '__main__':
    main('two_layer', TwoLayer, loss_fn, inputs=8, outputs=4)
