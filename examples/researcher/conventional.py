"""Conventional execution of the same ``ByteLM``/``objective``: eager PyTorch, or torch.compile + one CUDA graph.

* ``eager``: the model as written, ``torch.optim.AdamW(fused=True)``, global-norm clipping. One kernel per
  operation; the correctness reference for the first-update check (same explicit casts as the captured program).
* ``compiled``: Inductor-compiled forward/backward (``max-autotune-no-cudagraphs``), capturable fused AdamW, and the
  whole update (forward, loss, backward, clip, optimizer) replayed as one CUDA graph. This is the speed control:
  the fastest ordinary PyTorch recipe for the same arithmetic. If graph capture fails the arm records the error and
  runs the compiled update without a graph, labelled ``compiled-nograph``.

Both arms start from the ``ByteLM`` initialization for the config's seed, so they share their first parameters with
the Latch program captured from the same module.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from model import ByteLM, Config, adamw_hyperparameters, objective


class ConventionalUpdate:
    def __init__(self, cfg: Config, mode: str, device='cuda'):
        if mode not in ('eager', 'compiled'):
            raise ValueError('mode must be eager or compiled')
        self.cfg, self.mode, self.device = cfg, mode, torch.device(device)
        began = time.perf_counter()
        self.model = ByteLM(cfg).to(self.device)
        self.names = [name for name, _ in self.model.named_parameters()]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **adamw_hyperparameters(cfg),
                                           fused=True, capturable=(mode == 'compiled'))
        self.tokens = torch.zeros(cfg.batch, cfg.seq_len, dtype=torch.int64, device=self.device)
        self.labels = torch.zeros_like(self.tokens)
        self.aux_weight = torch.zeros((), dtype=torch.float32, device=self.device)
        self.loss = torch.zeros((), dtype=torch.float32, device=self.device)
        self.graph = None
        self.graph_error = None
        self._objective = objective_fn(self.model)
        if mode == 'compiled':
            self._objective = torch.compile(self._objective, mode='max-autotune-no-cudagraphs', dynamic=False)
            self._compile_and_capture()
        self.setup_seconds = time.perf_counter() - began

    # -- the update -------------------------------------------------------------------------------
    def _update(self):
        loss = self._objective(self.tokens, self.labels, self.aux_weight)
        loss.backward()
        if self.cfg.clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip, foreach=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=False)
        self.loss.copy_(loss.detach())

    def _compile_and_capture(self):
        initial = self.state_dict()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):                      # compile, autotune and allocate before capture
                self._update()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._update()
            self.graph = graph
        except Exception as error:                 # recorded, not hidden: the arm then runs compiled without a graph
            self.graph_error = repr(error)[-600:]
        torch.cuda.synchronize()
        self.load_state_dict(initial)               # warmup changed state; restore the fresh start in place

    def stage(self, tokens, labels, aux_weight):
        self.tokens.copy_(tokens, non_blocking=True)
        self.labels.copy_(labels, non_blocking=True)
        self.aux_weight.fill_(float(aux_weight))

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self._update()

    # -- persistent state ---------------------------------------------------------------------------
    @property
    def label(self):
        if self.mode == 'eager':
            return 'eager PyTorch: one kernel per operation, fused AdamW'
        if self.graph is not None:
            return 'torch.compile forward/backward + capturable fused AdamW, whole update replayed as one CUDA graph'
        return 'compiled-nograph: torch.compile forward/backward + fused AdamW, no CUDA graph (capture failed)'

    @property
    def arm(self):
        return self.mode if self.mode == 'eager' or self.graph is not None else 'compiled-nograph'

    def parameters(self):
        return dict(self.model.named_parameters())

    def state_dict(self):
        """CPU snapshot of parameters and AdamW state, keyed by parameter name."""
        optimizer = {}
        for name, parameter in self.model.named_parameters():
            state = self.optimizer.state.get(parameter, {})
            optimizer[name] = {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                               for key, value in state.items()}
        return {'arm': self.mode, 'config': self.cfg.to_dict(),
                'parameters': {name: p.detach().cpu().clone() for name, p in self.model.named_parameters()},
                'optimizer': optimizer}

    def load_state_dict(self, state):
        """Restore in place (pointers stay stable for a captured graph). Missing AdamW state means a fresh start."""
        if state['config'] != self.cfg.to_dict():
            raise ValueError('snapshot belongs to a different configuration')
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.copy_(state['parameters'][name])
                saved = state['optimizer'].get(name, {})
                current = self.optimizer.state.get(parameter)
                if current is None:
                    if saved:   # eager arm before its first update: adopt the saved AdamW state on this device
                        self.optimizer.state[parameter] = {
                            key: value.to(parameter.device).clone() if isinstance(value, torch.Tensor) else value
                            for key, value in saved.items()}
                    continue
                for key, value in current.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    if key in saved:
                        value.copy_(saved[key])
                    else:
                        value.zero_()                           # fresh moments and step counter
        torch.cuda.synchronize()


def objective_fn(model):
    def compute(tokens, labels, aux_weight):
        return objective(model(tokens), labels, aux_weight)
    return compute


@torch.no_grad()
def evaluate(cfg: Config, parameters, batches, device='cuda'):
    """Mean next-byte cross entropy of the final readout on evaluation batches, with the model as written.

    ``parameters`` maps canonical names to tensors from either execution path; evaluation arithmetic is the same
    eager forward for both, so dev numbers are comparable across arms.
    """
    if not batches:
        return None
    model = ByteLM(cfg).to(device)
    model.load_state_dict({name: value.detach().to(device) for name, value in parameters.items()}, strict=True)
    total, count = 0.0, 0
    for tokens, labels in batches:
        logits = model(tokens.to(device))[0]
        total += float(torch.nn.functional.cross_entropy(logits, labels.to(device).reshape(-1), reduction='sum'))
        count += labels.numel()
    nats = total / count
    return {'cross_entropy_nats': nats, 'bits_per_byte': nats / 0.6931471805599453, 'tokens': count}
