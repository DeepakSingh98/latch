"""Automatic batch-size variants behind one training entry point (release step 18).

``BatchVariants`` wraps the normal module capture/build path. The first update
at a new batch size captures the module at that batch, writes an ordinary
static artifact with ``CapturedModule.build`` and loads it with
``ModuleUpdate.load``. Returning to a batch size reuses that loaded update:
its compiled library, bound runtime, launch admission and arena are kept, so
no capture, lowering or compilation happens on a revisit.

Each variant is a separate compiled kernel. Persistent training state is not
per variant: parameters, optimizer moments/counters, explicit model/RNG state,
frozen buffers and a runtime learning rate are one canonical set of device
tensors, bound by pointer into every variant (``ModuleUpdate.load(bindings=)``).
A variant's kernel commits its update into those tensors, so switching batch
size continues the same training run without copies, resets or model copies.
Data inputs, hoisted constants and the arena remain private to each variant.

The supported contract is deliberately narrow: only the batch extent of each
data tensor may change. Sequence length, feature width, dtype, arity, module
structure, optimizer, precision and Python control flow are fixed when the
first variant is created; changes raise ``VariantContractError``. One kernel
that executes several batch sizes (release step 19) is a different design and
is not implemented here.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import types
import uuid

import torch

from .api import ModuleUpdate, _copy_bindings, _restore_copies, capture_module

STATE_SCHEMA = 'latch-batch-variants-state-v1'
PERSISTENT_ROLES = ('parameter', 'momentum', 'optimizer_state', 'state')


class VariantContractError(ValueError):
    """A request changes something other than the declared batch extent."""


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _source_sha(obj):
    try:
        return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
    except (OSError, TypeError):
        return None


def _qualname(obj):
    target = obj if inspect.isclass(obj) or inspect.isfunction(obj) else type(obj)
    return f'{getattr(target, "__module__", "?")}.{getattr(target, "__qualname__", repr(target))}'


def _tensor_bytes_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _layout(value):
    return {'shape': list(value.shape), 'strides': list(value.stride()), 'dtype': str(value.dtype)}


def _identity_value(value):
    """Stable configuration identity, including user-written update functions."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return {**_layout(value), 'sha256': _tensor_bytes_sha(value)}
    if isinstance(value, dict):
        return {str(key): _identity_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_identity_value(item) for item in value]
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, (set, frozenset)):
        return {'set': sorted((_identity_value(item) for item in value), key=_sha)}
    if isinstance(value, types.CodeType):
        # Exclude filenames/line numbers so independently installed workers
        # agree, but bind constants and nested comprehensions as well as opcodes.
        return {'bytecode': value.co_code.hex(), 'constants': _identity_value(value.co_consts),
                'names': value.co_names, 'variables': value.co_varnames,
                'freevars': value.co_freevars, 'cellvars': value.co_cellvars,
                'arguments': [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
                'flags': value.co_flags}
    if callable(value):
        code = getattr(value, '__code__', None)
        return {'callable': _qualname(value), 'source_sha256': _source_sha(value),
                'code': _identity_value(code),
                'defaults': _identity_value(getattr(value, '__defaults__', None)),
                'keyword_defaults': _identity_value(getattr(value, '__kwdefaults__', None)),
                'closure': [_identity_value(cell.cell_contents)
                            for cell in (getattr(value, '__closure__', None) or ())]}
    return {'type': _qualname(value), 'value': str(value)}


def persistent_keys(manifest, module_api):
    """Map flat argument indices to batch-independent state keys shared across variants."""
    keys = {}
    for index, name in enumerate(manifest['inputs']):
        role = manifest['input_roles'][name]
        if role['role'] in PERSISTENT_ROLES:
            keys[index] = (role['role'], role['parameter'])
    for name, index in module_api['input_indices']['buffers'].items():
        keys[index] = ('buffer', name)
    for name, index in module_api['input_indices'].get('frozen', {}).items():
        keys[index] = ('frozen', name)
    if 'optimizer_lr' in module_api['input_indices']:
        keys[module_api['input_indices']['optimizer_lr']] = ('optimizer_lr', '')
    return keys


@dataclass
class Variant:
    """One loaded batch specialization; its update owns only data, constants and arena."""
    key: tuple
    batch: int
    directory: Path
    update: ModuleUpdate
    identity: dict
    setup: dict
    updates: int = 0
    revisits: int = 0
    created_at_update: int = 0
    events: list = field(default_factory=list)

    def summary(self):
        runtime = self.update.runtime
        return {'batch': self.batch, 'directory': str(self.directory), 'identity': self.identity,
                'setup': self.setup, 'updates': self.updates, 'revisits': self.revisits,
                'created_at_update': self.created_at_update, 'runtime_id': id(runtime),
                'launch_count': runtime.launch_count, 'launch_cache_policy': dict(runtime.launch_cache_policy),
                'blocks': runtime.blocks, 'arena_bytes': runtime.arena.numel() * runtime.arena.element_size(),
                'runtime_allocated_bytes': runtime.allocated_bytes}


class BatchVariants:
    """Train one ordinary module through separately compiled batch-size variants.

    ``step(inputs, targets)`` accepts tuples of tensors, as ``capture_module``
    does. The first call at a batch size builds its variant; later calls reuse
    it. ``batch_dims`` gives the batch dimension of each input then target
    (default 0 for every tensor; ``None`` marks a batch-independent tensor).
    ``batch_sizes`` optionally restricts admitted batch sizes. At most
    ``max_variants`` variants stay loaded; the least recently used one is
    released beyond that bound (persistent state is unaffected; a later
    request rebuilds it, normally reusing the native build cache).
    """

    def __init__(self, model, loss_fn, *, work_dir, cache_dir, optimizer=None, optimizer_state=None,
                 state=None, precision=None, batch_sizes=None, batch_dims=None, max_variants=8,
                 device='cuda', arch='sm_90', blocks=None, retain_all=False, policy='auto',
                 precision_policy=None, compact_codegen='off', execution=None, compose_selection='auto',
                 dropout_seed=None, prepared_artifacts=None):
        if not isinstance(model, torch.nn.Module):
            raise TypeError('BatchVariants expects an nn.Module')
        if type(max_variants) is not int or max_variants < 1:
            raise ValueError('max_variants must be a positive integer')
        if batch_sizes is not None:
            batch_sizes = tuple(batch_sizes)
            if not batch_sizes or any(type(b) is not int or b <= 0 for b in batch_sizes):
                raise ValueError('batch_sizes must be positive integers')
        self.model, self.loss_fn = model, loss_fn
        self._optimizer = optimizer
        # Capture templates only: their values seed the first variant's state.
        self._optimizer_template = None if optimizer_state is None else {
            k: v.detach().clone() for k, v in optimizer_state.items()}
        self._state_template = None if state is None else {k: v.detach().clone() for k, v in state.items()}
        self._precision = precision
        self._dropout_seed = dropout_seed
        if prepared_artifacts is not None and (not isinstance(prepared_artifacts, dict)
                or any(type(batch) is not int or batch <= 0 for batch in prepared_artifacts)):
            raise VariantContractError('prepared_artifacts must map positive integer batches to directories')
        self._prepared_artifacts = (None if prepared_artifacts is None else
                                    {batch: Path(path) for batch, path in prepared_artifacts.items()})
        self.batch_sizes = None if batch_sizes is None else frozenset(batch_sizes)
        self._batch_dims = None if batch_dims is None else tuple(batch_dims)
        self.max_variants = max_variants
        self.device = torch.device(device)
        self.cache_dir = str(cache_dir)
        self.work_dir = Path(work_dir) / f'variants-{uuid.uuid4().hex[:12]}'
        self._build_options = dict(arch=arch, retain_all=retain_all, policy=policy,
                                   precision_policy=precision_policy, compact_codegen=compact_codegen,
                                   execution=execution, compose_selection=compose_selection)
        self._blocks = blocks
        self._variants = OrderedDict()
        self._active = self._last_key = None
        self._reference = None          # data-signature contract fixed by the first request
        self._contract = None           # persistent-state contract fixed by the first variant
        self._canonical = {}            # state key -> shared device tensor
        self._groups = None
        self.updates = 0
        self.batch_updates = {}
        self.counters = {'captures': 0, 'prepares': 0, 'native_compilations': 0, 'native_cache_hits': 0,
                         'loads': 0, 'switches': 0, 'revisits': 0, 'evictions': 0, 'rebuilds': 0}
        self.history = []
        self._seen = set()
        self._fingerprint = self._module_fingerprint()
        # Per-update fast checks: tracked tensor versions/storage, train flags and optimizer fields.
        # The full named walk is repeated whenever a variant is created.
        self._tracked = [t for _, t in model.named_parameters(remove_duplicate=False)]
        self._tracked += [t for _, t in model.named_buffers(remove_duplicate=False)]
        self._submodules = list(model.modules())
        self._quick_fingerprint = self._quick()
        self._optimizer_fields = None if optimizer is None else copy.deepcopy(vars(optimizer))
        self._memo = {}                 # validated request shapes -> (key, batch)
        self.identity = {
            'schema': 'latch-batch-variants-v1',
            'model': _qualname(model), 'model_source_sha256': _source_sha(type(model)),
            'loss': _qualname(loss_fn), 'loss_source_sha256': _source_sha(loss_fn),
            'parameters': [(n, list(p.shape), str(p.dtype)) for n, p in model.named_parameters(remove_duplicate=False)],
            'buffers': [(n, list(b.shape), str(b.dtype), _tensor_bytes_sha(b))
                        for n, b in model.named_buffers(remove_duplicate=False)],
            'optimizer': self._optimizer_identity(),
            'precision': None if precision is None else repr(precision),
            'dropout_seed': dropout_seed,
            'state': None if state is None else [(k, list(v.shape), str(v.dtype)) for k, v in state.items()],
            'build': {**self._build_options, 'blocks': blocks}, 'torch': torch.__version__,
            'batch_dims': self._batch_dims}
        self.identity_sha256 = _sha(self.identity)

    # ------------------------------------------------------------------ contract checks
    def _optimizer_identity(self):
        optimizer = self._optimizer
        if optimizer is None:
            return None
        return {'type': type(optimizer).__qualname__, 'fields': _identity_value(vars(optimizer))}

    def _prepared_identity(self):
        # A bare module artifact cannot prove which same-shaped Python model
        # generated it. The CPU preparation sidecar binds source/configuration
        # and initial tensor bytes, in addition to the native artifact hashes.
        implicit_rng = (int(torch.initial_seed()) if self._dropout_seed is None and any(
            isinstance(module, torch.nn.Dropout) and module.training and 0 < module.p < 1
            for module in self.model.modules()) else None)
        return {'manager': self.identity, 'implicit_dropout_seed': implicit_rng,
                'parameters': [(name, bool(value.requires_grad), _identity_value(value))
                               for name, value in self.model.named_parameters(remove_duplicate=False)],
                'buffers': [(name, _identity_value(value))
                            for name, value in self.model.named_buffers(remove_duplicate=False)],
                'state': _identity_value(self._state_template),
                'optimizer_state': _identity_value(self._optimizer_template),
                'loss': _identity_value(self.loss_fn),
                'modules': [(name, _qualname(module), _source_sha(type(module)),
                             {key: _identity_value(value) for key, value in vars(module).items()
                              if key not in {'_parameters', '_buffers', '_modules'}
                              and not 'hook' in key})
                            for name, module in self.model.named_modules()]}

    def _module_fingerprint(self):
        rows = [(n, p.data_ptr(), tuple(p.shape), p.dtype, p._version)
                for n, p in self.model.named_parameters(remove_duplicate=False)]
        rows += [(n, b.data_ptr(), tuple(b.shape), b.dtype, b._version)
                 for n, b in self.model.named_buffers(remove_duplicate=False)]
        return tuple(rows), self.model.training

    def _quick(self):
        return (tuple((t._version, t.data_ptr()) for t in self._tracked),
                tuple(m.training for m in self._submodules))

    def _check_module(self, full=True):
        if self._quick() != self._quick_fingerprint or (full and self._module_fingerprint() != self._fingerprint):
            raise VariantContractError(
                'module parameters, buffers or train/eval mode changed after BatchVariants was created; '
                'its training state lives in BatchVariants (use state_dict/load_state_dict), and structural '
                'changes need a new BatchVariants')
        if self._optimizer is not None and vars(self._optimizer) != self._optimizer_fields:
            raise VariantContractError('optimizer configuration changed; hyperparameters are compiled into every '
                                       'variant (use set_learning_rate for a runtime AdamW rate)')

    def signature(self, inputs, targets=()):
        """Validate a request against the contract and return (key, batch)."""
        if not isinstance(inputs, (tuple, list)) or not isinstance(targets, (tuple, list)):
            raise TypeError('inputs and targets must be tuples/lists of tensors')
        tensors = (*inputs, *targets)
        if not tensors or any(not isinstance(t, torch.Tensor) for t in tensors):
            raise VariantContractError('every input and target must be a tensor')
        dims = self._batch_dims if self._batch_dims is not None else (0,) * len(tensors)
        if len(dims) != len(tensors):
            raise VariantContractError(f'batch_dims declares {len(dims)} tensors; request has {len(tensors)}')
        batch = None
        rows = []
        for index, (tensor, dim) in enumerate(zip(tensors, dims)):
            shape = list(tensor.shape)
            if dim is not None:
                if not 0 <= dim < tensor.ndim:
                    raise VariantContractError(f'tensor {index} has no batch dimension {dim}')
                if batch is None:
                    batch = shape[dim]
                elif shape[dim] != batch:
                    raise VariantContractError(f'tensor {index} batch extent {shape[dim]} differs from {batch}; '
                                               'all batched tensors must share one batch size')
                shape[dim] = None
            rows.append((tuple(shape), str(tensor.dtype)))
        if batch is None:
            raise VariantContractError('no tensor carries a batch dimension')
        if type(batch) is not int or batch <= 0:
            raise VariantContractError('batch size must be a positive integer')
        reference = (len(inputs), len(targets), tuple(rows))
        if self._reference is not None and reference != self._reference:
            arity = (len(inputs), len(targets)) != self._reference[:2]
            detail = 'input/target arity' if arity else next(
                f'tensor {i}: {"dtype" if a[1] != b[1] else "non-batch shape"} {b} -> {a}'
                for i, (a, b) in enumerate(zip(rows, self._reference[2])) if a != b)
            raise VariantContractError(
                f'only the batch extent may change between variants ({detail}); sequence length, width, dtype '
                'and arity changes are outside the batch-variant contract')
        if self.batch_sizes is not None and batch not in self.batch_sizes:
            raise VariantContractError(f'batch size {batch} is not in the declared set {sorted(self.batch_sizes)}')
        if self._reference is None:
            self._reference = reference
        return (batch, reference), batch

    # ------------------------------------------------------------------ variant construction
    def _state_contract(self, manifest, receipt, keys, saved):
        layouts = {f'{role}:{name}': _layout(saved[index]) for index, (role, name) in keys.items()}
        buffers = {f'{role}:{name}': _tensor_bytes_sha(saved[index])
                   for index, (role, name) in keys.items() if role in {'buffer', 'frozen'}}
        dropout = receipt['module_api'].get('dropout')
        if dropout is not None:
            # A variant changes the activation extent and Philox block count.
            # The RNG binding, dropout policy and call order must remain shared.
            dropout = {**{key: value for key, value in dropout.items() if key != 'calls'},
                       'call_sites': [call['site'] for call in dropout.get('calls', ())]}
        return {'schema': STATE_SCHEMA, 'layouts': layouts, 'buffers': buffers,
                'optimizer_state_schema': manifest.get('optimizer_state_schema', []),
                'optimizer': receipt['optimizer'], 'parameter_aliases': receipt['parameter_aliases'],
                'buffer_aliases': receipt['module_api']['buffer_aliases'],
                'frozen_aliases': receipt['module_api'].get('frozen_aliases', {}),
                'dropout': dropout,
                'stateful_forward': receipt['module_api']['stateful_forward']}

    def _build_artifact(self, batch, inputs, targets, directory):
        started = time.perf_counter()
        # Values do not affect the captured program; zeros keep data out of artifacts.
        def example(t):
            return torch.zeros(t.shape, dtype=t.dtype)
        captured = capture_module(self.model, self.loss_fn, tuple(map(example, inputs)),
                                  tuple(map(example, targets)), optimizer=self._optimizer,
                                  optimizer_state=self._optimizer_template, state=self._state_template,
                                  precision=self._precision, dropout_seed=self._dropout_seed)
        captured_at = time.perf_counter()
        self.counters['captures'] += 1
        # Native builds live in <cache_dir>/<build key>/; a key present before this request is a cache hit.
        cache_root = Path(self.cache_dir)
        if self.cache_dir == '/out/cache' and os.environ.get('LATCH_OPS_BUILD_CACHE'):
            cache_root = Path(os.environ['LATCH_OPS_BUILD_CACHE'])   # runtime.build's worker redirection
        cached_keys = set(p.name for p in cache_root.iterdir()) if cache_root.is_dir() else set()
        receipt = captured.build(directory, cache_dir=self.cache_dir, **self._build_options)
        prepared_at = time.perf_counter()
        self.counters['prepares'] += 1
        library = Path(receipt['library'])
        compiled = library.parent.name not in cached_keys
        self.counters['native_compilations' if compiled else 'native_cache_hits'] += 1
        return receipt, {'capture_seconds': captured_at - started,
                         'prepare_seconds': prepared_at - captured_at,
                         'build_seconds': receipt['build_seconds'], 'native_compiled': compiled}

    def prepare_artifact(self, inputs, targets=()):
        """Capture/build one variant on CPU, returning its portable artifact path.

        No Runtime or CUDA tensor is created. Reconstruct the same seeded
        model/loss/optimizer/state on the device worker and pass the returned
        paths as ``prepared_artifacts={batch: path}``. That mode never captures
        or compiles, even when an evicted variant is loaded again. Keep the
        native library at the absolute path recorded by the normal build API.
        """
        if self._prepared_artifacts is not None:
            raise VariantContractError('prepared-artifact managers are load-only')
        tensors = [*self.model.parameters(), *self.model.buffers(), *inputs, *targets,
                   *(self._state_template or {}).values(), *(self._optimizer_template or {}).values()]
        if self.device.type != 'cpu' or any(value.device.type != 'cpu' for value in tensors):
            raise VariantContractError('prepare_artifact requires device="cpu" and CPU model/state/data')
        self._check_module(full=True)
        key, batch = self.signature(inputs, targets)
        directory = self.work_dir / f'prepared-b{batch}'
        receipt, setup = self._build_artifact(batch, tuple(inputs), tuple(targets), directory)
        record = {'schema': 'latch-prepared-batch-variant-v1', 'batch': batch,
                  'signature': [batch, json.loads(json.dumps(key[1]))],
                  'manager_sha256': _sha(self._prepared_identity()),
                  'artifact_sha256': hashlib.sha256((directory / 'artifact.json').read_bytes()).hexdigest(),
                  'source_hash': receipt['source_hash'], 'setup': setup}
        (directory / 'batch-variant.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        return directory

    def _read_prepared(self, key, batch):
        directory = self._prepared_artifacts.get(batch)
        if directory is None:
            raise VariantContractError(f'batch {batch} has no prepared artifact; compilation is disabled')
        record = json.loads((directory / 'batch-variant.json').read_text())
        if (record.get('schema') != 'latch-prepared-batch-variant-v1' or record.get('batch') != batch
                or record.get('signature') != [batch, json.loads(json.dumps(key[1]))]
                or record.get('manager_sha256') != _sha(self._prepared_identity())):
            raise VariantContractError(f'batch {batch} prepared model/loss/optimizer/state/signature differs')
        raw = (directory / 'artifact.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.get('artifact_sha256'):
            raise VariantContractError(f'batch {batch} prepared artifact receipt changed')
        receipt = json.loads(raw)
        # Verify data before using its manifest or serialized CPU bindings to
        # select shared state. ModuleUpdate.load independently verifies again.
        for name, expected in receipt['files'].items():
            if (Path(name).name != name
                    or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected):
                raise VariantContractError(f'batch {batch} prepared artifact file changed: {name}')
        if record.get('source_hash') != receipt.get('source_hash'):
            raise VariantContractError(f'batch {batch} prepared source identity changed')
        return directory, receipt

    def _create(self, key, batch, inputs, targets):
        started = time.perf_counter()
        if self._prepared_artifacts is None:
            directory = self.work_dir / f'b{batch}-{len(self.history)}'
            receipt, setup = self._build_artifact(batch, inputs, targets, directory)
        else:
            directory, receipt = self._read_prepared(key, batch)
            setup = {'capture_seconds': 0., 'prepare_seconds': 0., 'build_seconds': 0.,
                     'native_compiled': False, 'prepared_artifact': True}
        prepared_at = time.perf_counter()
        library = Path(receipt['library'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        saved = torch.load(directory / 'bindings.pt', weights_only=True, map_location='cpu')['args']
        keys = persistent_keys(manifest, receipt['module_api'])
        contract = self._state_contract(manifest, receipt, keys, saved)
        if self._contract is None:
            update = ModuleUpdate.load(directory, device=self.device, blocks=self._blocks)
            self._canonical = {k: update.runtime.args[index] for index, k in keys.items()}
            self._contract = contract
            self.contract_sha256 = _sha(contract)
        else:
            if contract != self._contract:
                changed = sorted(k for k in set(contract) | set(self._contract)
                                 if contract.get(k) != self._contract.get(k))
                raise VariantContractError(f'batch {batch} changes the persistent training-state contract '
                                           f'({", ".join(changed)}); only data extents may vary')
            bindings = {index: self._canonical[k] for index, k in keys.items()}
            update = ModuleUpdate.load(directory, device=self.device, blocks=self._blocks, bindings=bindings)
        loaded_at = time.perf_counter()
        self.counters['loads'] += 1
        runtime = update.runtime
        if self._groups is None:
            self._groups = {kind: dict(getattr(runtime, kind)) for kind in ('parameters', 'optimizer_state', 'state')}
        identity = {'manager_sha256': self.identity_sha256, 'contract_sha256': self.contract_sha256,
                    'signature': [batch, json.loads(json.dumps(key[1]))], 'source_hash': receipt['source_hash'],
                    'library_sha256': receipt['library_sha256'], 'build_key': library.parent.name,
                    'arch': receipt['arch'], 'torch': receipt['torch'],
                    'execution': receipt['execution'],
                    'execution_selection': receipt.get('execution_selection', {}).get('selected'),
                    'immutable_bindings_sha256': receipt['immutable_bindings_sha256']}
        identity['variant_key_sha256'] = _sha(identity)
        setup.update(load_seconds=loaded_at - prepared_at, total_seconds=loaded_at - started)
        variant = Variant(key, batch, directory, update, identity, setup, created_at_update=self.updates)
        if key in self._seen:
            self.counters['rebuilds'] += 1
        self._seen.add(key)
        self.history.append({'event': 'created', 'batch': batch, 'update': self.updates,
                             'source_hash': receipt['source_hash'], **setup})
        self._variants[key] = variant
        self._evict()
        return variant

    def _evict(self):
        # The newest entry is about to become active; release least recently used ones.
        # Shared state tensors are held by this manager, never by an individual variant.
        while len(self._variants) > self.max_variants:
            variant = self._variants.pop(next(iter(self._variants)))
            if variant is self._active:
                self._active = None
            self.history.append({'event': 'evicted', 'batch': variant.batch, 'update': self.updates})
            self.counters['evictions'] += 1
            variant.update = None

    def _select(self, inputs, targets):
        self._check_module(full=False)
        try:
            quick = (len(inputs), tuple((t.shape, t.dtype) for t in inputs), tuple((t.shape, t.dtype) for t in targets))
        except (AttributeError, TypeError):
            quick = None
        hit = self._memo.get(quick) if quick is not None else None
        if hit is None:
            hit = self.signature(inputs, targets)       # full validation; raises on contract changes
            if quick is not None and len(self._memo) < 64:
                self._memo[quick] = hit
        key, batch = hit
        variant = self._variants.get(key)
        if variant is None:
            self._check_module(full=True)
            variant = self._create(key, batch, inputs, targets)
        else:
            self._variants.move_to_end(key)
            if self._active is not variant:
                variant.revisits += 1
                self.counters['revisits'] += 1
        if self._last_key is not None and self._last_key != key:
            self.counters['switches'] += 1
        self._active, self._last_key = variant, key
        return variant

    # ------------------------------------------------------------------ training entry point
    def prepare(self, inputs, targets=()):
        """Build or select the variant for this batch without updating."""
        return self._select(tuple(inputs), tuple(targets))

    def step(self, inputs, targets=()):
        """One complete native update at this batch size; returns the variant's loss view."""
        inputs, targets = tuple(inputs), tuple(targets)
        variant = self._select(inputs, targets)
        variant.update.stage(inputs, targets)
        variant.update.step()
        variant.updates += 1
        self.updates += 1
        self.batch_updates[variant.batch] = self.batch_updates.get(variant.batch, 0) + 1
        return variant.update.runtime.loss

    def variant(self, batch):
        """The loaded variant for a batch size (or None)."""
        return next((v for v in self._variants.values() if v.batch == batch), None)

    @property
    def active(self):
        return self._active

    def _require_state(self):
        if self._groups is None:
            raise RuntimeError('no variant exists yet; call prepare(inputs, targets) or step() first')

    @property
    def parameters(self):
        self._require_state()
        return self._groups['parameters']

    @property
    def optimizer_state(self):
        self._require_state()
        return self._groups['optimizer_state']

    @property
    def state(self):
        self._require_state()
        return self._groups['state']

    def set_learning_rate(self, value):
        """Stage the shared runtime AdamW rate used by every variant."""
        self._require_state()
        next(reversed(self._variants.values())).update.set_learning_rate(value)

    def state_dict(self):
        """Synchronized CPU snapshot of the logical training state (batch independent)."""
        self._require_state()
        result = {'schema': STATE_SCHEMA, 'contract_sha256': self.contract_sha256,
                  'updates': self.updates, 'batch_updates': dict(self.batch_updates)}
        for kind, group in self._groups.items():
            result[kind] = {name: value.detach().cpu().clone() for name, value in group.items()}
        return result

    def load_state_dict(self, state):
        """Restore parameters, optimizer and model/RNG state into the shared bindings."""
        self._require_state()
        if state.get('schema') != STATE_SCHEMA:
            raise ValueError('not a batch-variant training-state snapshot')
        if state.get('contract_sha256') != self.contract_sha256:
            raise ValueError('snapshot belongs to a different module/optimizer/state contract')
        updates = state.get('updates')
        if type(updates) is not int or updates < 0:
            raise ValueError('snapshot update count must be a nonnegative integer')
        _copy_bindings(_restore_copies(next(reversed(self._variants.values())).update.runtime, state))
        self.updates = updates
        self.batch_updates = {int(k): int(v) for k, v in state.get('batch_updates', {}).items()}

    def stats(self):
        """Counters, events and loaded-variant summaries (no device synchronization)."""
        return {'identity_sha256': self.identity_sha256,
                'contract_sha256': getattr(self, 'contract_sha256', None),
                'updates': self.updates, 'batch_updates': dict(self.batch_updates),
                'counters': dict(self.counters), 'max_variants': self.max_variants,
                'loaded': [v.summary() for v in self._variants.values()],
                'history': [dict(row) for row in self.history]}

    def memory(self):
        """Bytes of shared state versus per-variant private bindings and arenas."""
        shared = {id(t): t.numel() * t.element_size() for t in self._canonical.values()}
        rows = {}
        for v in self._variants.values():
            runtime = v.update.runtime
            private = {id(t): t.numel() * t.element_size() for t in runtime.args if id(t) not in shared}
            rows[v.batch] = {'private_binding_bytes': sum(private.values()),
                             'runtime_allocated_bytes': runtime.allocated_bytes}
        return {'shared_state_bytes': sum(shared.values()), 'variants': rows,
                'total_bytes': sum(shared.values()) + sum(r['private_binding_bytes'] + r['runtime_allocated_bytes']
                                                          for r in rows.values())}
