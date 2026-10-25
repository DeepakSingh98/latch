"""Residual module, auxiliary head and explicit advancing Philox dropout."""
from __future__ import annotations

import torch

from dynamic_batch_example import main
from megakernel_lab.operation_compiler.random import dropout, make_rng_state


class ResidualDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(16, 32)
        self.residual = torch.nn.Linear(32, 32)
        self.head = torch.nn.Linear(32, 8)
        self.auxiliary = torch.nn.Linear(32, 8)

    def forward(self, x, *, state):
        hidden = torch.tanh(self.hidden(x))
        hidden = hidden + .25 * torch.tanh(self.residual(hidden))
        hidden, rng = dropout(hidden, state['rng'], p=.25)
        return (self.head(hidden), self.auxiliary(hidden)), {'rng': rng}


def loss_fn(prediction, target):
    main_prediction, auxiliary_prediction = prediction
    return ((main_prediction - target).square().mean()
            + .15 * (auxiliary_prediction - target).square().mean())


def initial_state():
    return {'rng': make_rng_state(0x123456789ABCDEF0, counter=(1 << 32) - 17)}


if __name__ == '__main__':
    main('residual_dropout', ResidualDropout, loss_fn, inputs=16, outputs=8,
         state_factory=initial_state)
