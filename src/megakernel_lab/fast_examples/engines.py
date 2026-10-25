"""Conventional control engines and the native engine for module-API families.

Arms:
  eager          ordinary PyTorch autograd + torch.optim (fused where available), uncaptured; numerical calibration
  eager-graph    the same eager step captured whole in one CUDA graph (static input/grad buffers)
  compiled*      torch.compile(fullgraph, max-autotune-no-cudagraphs) of a functional gradient + optimizer update,
                 then the whole update captured in one CUDA graph; gradients are compiled outputs (no extra copies)
  latch*         ModuleUpdate.load of a CPU-prepared release artifact; step() captured in one CUDA graph
  latch-direct   the same artifact, step() launched directly (public default use, no CUDA graph)
No arm silently falls back: a compile/capture failure fails the arm.
"""
import torch


def _cp(d):
    return {n: v.detach().cpu().clone() for n, v in d.items()}


class Engine:
    compiled = False
    graph = None
    label = ''

    def finite(self):
        return all(bool(torch.isfinite(v.float()).all()) for v in self.params.values())


class Conventional(Engine):
    """family: object with make_model(c, arm), objective(model, batch, c), make_optimizer(c, params, arm),
    bind_state(c, opt, params, state), state_names, functional_update(c, params, grads, state)."""

    def __init__(self, family, c, arm, example_batch):
        self.F, self.c, self.arm = family, c, arm
        self.model = family.make_model(c, arm).cuda()
        self.params = dict(self.model.named_parameters())
        self.opt_state = family.initial_state(c, self.params)
        self.batch = tuple(t.clone() for t in example_batch)
        self.loss = torch.zeros((), device='cuda')
        if arm.startswith('compiled'):
            self._compile()
        else:
            self.opt = family.make_optimizer(c, list(self.params.values()), arm)
            family.bind_state(c, self.opt, self.params, self.opt_state)
            self.grads = {n: torch.zeros_like(p) for n, p in self.params.items()}
            if arm == 'eager-graph':
                self._capture_eager()
        self.label = family.arm_label(c, arm)

    # --- eager / eager-graph -------------------------------------------------
    def _eager_step(self):
        loss = self.F.objective(self.model, self.batch, self.c)
        loss.backward()
        if hasattr(self.F, 'pre_step'):
            self.F.pre_step(self.c, list(self.params.values()))
        self.opt.step()
        return loss

    def _capture_eager(self):
        initial = self.state()
        for p in self.params.values():
            p.grad = torch.zeros_like(p)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for p in self.params.values():
                    p.grad.zero_()
                self._eager_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for p in self.params.values():
                p.grad.zero_()
            self.static_loss = self._eager_step()
        self.grads = {n: p.grad for n, p in self.params.items()}
        self.loss = self.static_loss
        self.load(initial)

    # --- compiled ------------------------------------------------------------
    def _compile(self):
        from torch.func import functional_call, grad_and_value
        F, c, model = self.F, self.c, self.model

        def compute_loss(params, batch):
            return F.objective_functional(lambda *a: functional_call(model, params, a), batch, c)
        derivative = grad_and_value(compute_loss)
        # Detached views share storage with the module parameters: the compiled update is an inference graph
        # whose in-place writes land in the real parameters (HARNESS-02 repair: requires_grad leaves made
        # AOTAutograd trace a backward through the in-place update).
        params = {n: p.detach() for n, p in self.params.items()}
        state, batch = self.opt_state, self.batch

        def update():
            grads, loss = derivative(params, batch)
            with torch.no_grad():
                F.functional_update(c, params, grads, state)
            return loss, grads
        mode = None if self.arm.endswith('-default') else 'max-autotune-no-cudagraphs'   # historical step-14 control
        self.fn = torch.compile(update, fullgraph=True, mode=mode, dynamic=False)
        initial = self.state()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.grads = self.fn()
        self.compiled = True
        self.load(initial)

    # --- common --------------------------------------------------------------
    def stage(self, batch):
        for dst, src in zip(self.batch, batch):
            dst.copy_(src)

    def step(self):
        if self.graph is not None:
            self.graph.replay()
            return
        self.opt.zero_grad(set_to_none=False)
        loss = self._eager_step_record()
        self.loss = loss.detach()

    def _eager_step_record(self):
        loss = self.F.objective(self.model, self.batch, self.c)
        loss.backward()
        with torch.no_grad():
            for n, p in self.params.items():
                self.grads[n].copy_(p.grad)      # raw gradients, before any optimizer-side clipping
        if hasattr(self.F, 'pre_step'):
            self.F.pre_step(self.c, list(self.params.values()))
        self.opt.step()
        return loss

    def state(self):
        return dict(parameters=_cp(self.params), optimizer=_cp(self.opt_state))

    def load(self, s):
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(s['parameters'][n])
            for n, v in self.opt_state.items():
                v.copy_(s['optimizer'][n])


class Native(Engine):
    def __init__(self, directory, graph=True, label='latch'):
        from megakernel_lab.operation_compiler import ModuleUpdate
        self.u = ModuleUpdate.load(directory)
        self.params = self.u.parameters
        self.opt_state = self.u.runtime.optimizer_state
        self.grads = self.u.grads
        self.loss = self.u.loss
        self.label = label
        if graph:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.u.step()

    def stage(self, batch):
        inputs, targets = self.split(batch)
        self.u.stage(inputs, targets)

    def split(self, batch):
        return batch[:1], batch[1:]

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.u.step()

    def state(self):
        return dict(parameters=_cp(self.params), optimizer=_cp(self.opt_state))

    def load(self, s):
        t = self.u.state_dict()
        t['parameters'] = {n: s['parameters'][n].to(dtype=t['parameters'][n].dtype) for n in t['parameters']}
        t['optimizer_state'] = {n: s['optimizer'][n].to(dtype=t['optimizer_state'][n].dtype)
                                for n in t['optimizer_state']}
        self.u.load_state_dict(t)


class NativeRuntime(Engine):
    """A CPU-prepared lower()/build() program (capture_training/loop/sampled path) bound with Runtime.

    Directory layout written by write_program(): program files + bindings.pt + receipt.json.
    Data inputs are the manifest inputs whose role is not parameter/optimizer_state/state/constant.
    """
    def __init__(self, directory, graph=True, label='latch', n_inputs=None):
        import json
        from pathlib import Path
        from megakernel_lab.operation_compiler.lower import GeneratedProgram
        from megakernel_lab.operation_compiler.runtime import Runtime
        directory = Path(directory)
        receipt = json.loads((directory / 'receipt.json').read_text())
        program = GeneratedProgram.load(directory)
        if program.cache_key != receipt['source_hash']:
            raise ValueError('native source differs from the CPU build receipt')
        saved = torch.load(directory / 'bindings.pt', weights_only=True, map_location='cpu')
        args = []
        for value in saved:
            bound = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device='cuda')
            bound.copy_(value)
            args.append(bound)
        self.rt = Runtime(program, tuple(args), library=receipt['library'])
        roles = self.rt.manifest['input_roles']
        names = self.rt.manifest['inputs']
        self.data_positions = [i for i, n in enumerate(names) if roles[n]['role'] == 'input']   # HARNESS-03 repair
        self.roles = {n: roles[n]['role'] for n in names}
        if n_inputs is not None and len(self.data_positions) != n_inputs:
            raise ValueError(f'expected {n_inputs} data inputs, roles {self.roles}')
        self.params = self.rt.parameters
        self.opt_state = self.rt.optimizer_state
        self.model_state = self.rt.state
        self.grads = self.rt.grads
        self.loss = self.rt.loss
        self.label = label
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            snapshot = self.state()
            with torch.cuda.stream(stream):
                self.rt.step()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.rt.step()
            self.load(snapshot)

    def stage(self, batch):
        for position, value in zip(self.data_positions, batch):
            self.rt.args[position].copy_(value)

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.rt.step()

    def state(self):
        s = dict(parameters=_cp(self.params), optimizer=_cp(self.opt_state))
        if self.model_state:
            s['model_state'] = _cp(self.model_state)
        return s

    def load(self, s):
        with torch.no_grad():
            for n, p in self.params.items():
                p.copy_(s['parameters'][n])
            for n, v in self.opt_state.items():
                v.copy_(s['optimizer'][n])
            for n, v in self.model_state.items():
                if 'model_state' in s:
                    v.copy_(s['model_state'][n])


def write_program(program, captured, directory, cache_dir, extra=None):
    """CPU side of the lower()/build() path used by the release's shipped runtime-loop example."""
    import hashlib
    import json
    import time
    from pathlib import Path
    from megakernel_lab.operation_compiler.runtime import build
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    program.write(directory)
    torch.save(tuple(v.detach().clone() for v in captured.example_args), directory / 'bindings.pt')
    t = time.perf_counter()
    library = build(program, cache_dir=str(cache_dir), arch='sm_90')
    receipt = dict(source_hash=program.cache_key, library=str(library),
                   library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                   build_seconds=time.perf_counter() - t,
                   execution=program.manifest.get('execution_plan', {}).get('selected', 'legacy'),
                   execution_selection=program.manifest.get('execution_selection'), **(extra or {}))
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    return receipt
