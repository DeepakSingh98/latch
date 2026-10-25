# Provenance: copied from the Latch repository, commit df184a5f2579
# (branch codex/latch-gradlev), research/latch_gradlev/gl/latch_run.py (git blob 240538b09fc668455f6088113c51ab223bbfdb61).
# Changes: none to the code (byte-identical below this header); imports megakernel_lab.operation_compiler.inference from the installed wheel.
"""Runtime wrapper for a prepared Latch loop artifact (built on a CPU worker by latch_build)."""
from __future__ import annotations

import torch


class LatchLoop:
    def __init__(self, artifact, c, W1_0, W2_0, B, T, device='cuda'):
        from megakernel_lab.operation_compiler.inference import Inference
        self.inf = Inference.load(artifact, device=device)
        self.B, self.T = B, T
        self.inf.load_parameters({k: c[k] for k in self.inf.parameters})
        self.W1_0, self.W2_0 = W1_0, W2_0
        self.X, self.U, self.TGT, self.E1, self.E2 = self.inf.inputs
        self.state = self.inf.state
        self.inf.set_count(T)

    def load(self, u, x, eta1, eta2, targets):
        self.X.copy_(x.transpose(0, 1))
        self.U.copy_(u.transpose(0, 1))
        self.E1.copy_(eta1.transpose(0, 1))
        self.E2.copy_(eta2.transpose(0, 1))
        self.TGT.copy_(targets.transpose(0, 1))

    def reset(self):
        self.state['W1'].copy_(self.W1_0.expand_as(self.state['W1']))
        self.state['W2'].copy_(self.W2_0.expand_as(self.state['W2']))

    def run_block(self, reset=True, count=None):
        if reset:
            self.reset()
        if count is not None:
            self.inf.set_count(count)
        out = self.inf.run()
        return out

    @property
    def W1(self):
        return self.state['W1']

    @property
    def W2(self):
        return self.state['W2']
