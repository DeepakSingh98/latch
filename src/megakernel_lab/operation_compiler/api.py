"""Small module entry point for the research operation compiler.

Capture and native compilation belong on a CPU build worker. Loading a prepared
artifact and stepping it belong on a CUDA worker. No model-family selection or
conventional fallback is hidden behind this API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.func import functional_call

from .capture import SGDConfig, capture_training
from .dynamic_batch import DynamicBatch, capture_dynamic_module
from .ir import CaptureError


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _storage_key(value):
    if not isinstance(value, torch.Tensor) or not value.numel():
        return None
    return str(value.device), value.untyped_storage().data_ptr()


def _copy_bindings(copies):
    """Apply simultaneous value copies, including swaps of existing bindings."""
    destinations = {_storage_key(target) for target, _ in copies} - {None}
    sources = [(target, source.detach().clone() if _storage_key(source) in destinations else source)
               for target, source in copies]
    with torch.no_grad():
        for target, source in sources:
            target.copy_(source)


def _restore_copies(runtime, state):
    """Validated (binding, saved value) pairs for persistent parameter/optimizer/model state."""
    schema = {entry['name']: entry['kind'] for entry in runtime.manifest.get('optimizer_state_schema', ())}
    copies = []
    for kind in ('parameters', 'optimizer_state', 'state'):
        current, saved = getattr(runtime, kind), state.get(kind, {})
        if set(current) != set(saved):
            raise ValueError(f'{kind} state keys differ')
        for name, value in current.items():
            source = saved[name]
            if not isinstance(source, torch.Tensor) or source.shape != value.shape or source.dtype != value.dtype:
                raise ValueError(f'{kind}:{name} state shape/dtype differs')
            if kind == 'optimizer_state':
                field = schema.get(name)
                if field == 'step' and not 0 <= int(source.item()) < 2**24:
                    raise ValueError('AdamW restored steps must be in the supported [0, 2**24) range')
                if field in {'exp_avg', 'exp_avg_sq'} and not bool(torch.isfinite(source).all()):
                    raise ValueError('AdamW restored moments must be finite')
                if field == 'exp_avg_sq' and bool((source < 0).any()):
                    raise ValueError('AdamW restored second moments must be nonnegative')
            copies.append((value, source))
    return copies


def _immutable_identity(args, manifest, module_api):
    """Bind frozen values independently of initial parameters and minibatches."""
    indices = set(module_api['input_indices']['buffers'].values())
    # Release step 07: frozen parameters are immutable bindings too. Artifacts
    # without frozen parameters keep their previous identity.
    indices.update(module_api['input_indices'].get('frozen', {}).values())
    indices.update(index for index, name in enumerate(manifest['inputs'])
                   if manifest['input_roles'][name]['role'] == 'constant_input')
    digest = hashlib.sha256()
    for index in sorted(indices):
        value = args[index]
        descriptor = {'index': index, 'shape': list(value.shape),
                      'strides': list(value.stride()), 'dtype': str(value.dtype)}
        digest.update(json.dumps(descriptor, sort_keys=True).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class CapturedModule:
    captured: object
    input_indices: dict
    buffer_aliases: dict

    def __getattr__(self, name):
        if name == 'captured':
            raise AttributeError(name)
        return getattr(self.captured, name)

    def build(self, directory, *, cache_dir, arch='sm_90', retain_all=False, policy='auto',
              precision_policy=None, compact_codegen='off', execution=None, compose_selection='auto',
              tuning=None, gradient_outputs='returned', fast_build=False, fast_device_compile=False,
              native_code_only=False, disable_expensive_optimizations=False, ptxas_optimization_level=None,
              recurrent_gradients='inline',
              explain=False, bf16_weight_chunk_k=None, square_sum_association=None, device_units=None,
              device_unit_inline='wgmma', device_unit_cache=None, device_unit_jobs=None, device_unit_split=None,
              device_unit_shared_pool=False, static_shared_reserve_bytes=0, update_schedule=None,
              segmented_products=None, norm_fusion=None, attention_rules=None, matrix_schedule=None,
              accumulation_rules=None, memory_rules=None, matrix_bodies=None, exact_bf16_products=None):
        """Write a bound native artifact; calls nvcc only on the build worker.

        explain=True additionally writes ``<directory>/explain/report.{json,md}`` (plus the captured IR
        and collector diagnostics) describing selections, rejections, copies/casts and structure. It
        changes no generated byte: the report is produced from the manifest after lowering.

        ``tuning=ops.MatrixTuning(cache=...)`` (opt-in, GPU worker) selects among existing matrix bodies by
        bounded complete-update measurement and caches the choice; see ``ops.matrix_tuning``.
        gradient_outputs='consumed' omits public gradients after the optimizer consumes them.
        fast_build=True uses the qualified single-unit driver shim, with the ordinary build as fallback.
        fast_device_compile=True separately requests nvcc's maximum device compilation
        preference; changed binaries require numerical and runtime qualification.
        native_code_only=True omits PTX fallback images, retaining only the
        selected real GPU target and the ordinary CUDA host launcher.
        disable_expensive_optimizations=True disables ptxas's memory-intensive
        optimization passes; changed binaries need independent qualification.
        ptxas_optimization_level optionally selects assembler level 0..3;
        omission preserves defaults and changed binaries need qualification.
        device_units=K compiles whole publication intervals as separately compiled device
        functions in K concurrent relocatable units behind the one entry (runtime.build);
        changed binaries need numerical and runtime qualification.
        static_shared_reserve_bytes reserves part of the common-compose shared-memory
        budget for linked static storage; zero preserves the existing planner.
        recurrent_gradients='deferred' opts into the measured loop speed/memory tradeoff.
        update_schedule='streamed' opts into ops.update_schedule's structural interval rules
        (single-reader chain sinking, barrier-free smem-free neighbours, rotated grid-stride
        starts) for flat common compose programs; None leaves lowering unchanged.
        update_schedule='streamed_prep' adds ops.prologue_batch's table phases for runs of
        tiny independent prologue phases (optimizer scalars, reduction-counter resets).
        segmented_products='flat' opts into the flattened segmented (expert) product body of
        ops.ragged_experts_lower (bitwise the default body's results); None leaves lowering unchanged.
        norm_fusion='rows' opts into ops.norm_fusion's structural normalization/activation/residual
        rules for flat common compose programs (coalesced exact norm-weight column sums, recomputed
        normalized rows and gated activations, residual-addend row prologues); None leaves lowering
        unchanged.
        attention_rules='streamed' opts into ops.attention_rules' structural preparation/layout
        rules for composed explicit-BF16 flat-long attention (shared mask tables, in-item key
        norms, contiguity-clone removal); None leaves the attention tape unchanged.
        matrix_schedule='underfilled' opts into ops.matrix_schedule's rules for underfilled long-K
        TMA products (deeper operand rings; under practical precision, deterministic split-K with an
        ascending FP32 slice sum); None leaves lowering unchanged.
        matrix_bodies='exact' (or a '+'-joined subset of tailrot, taildeep, fill, p5c, isolate) opts into
        ops.matrix_bodies' bitwise non-expert matrix body rules (rotated placement and deeper rings of mma.sync
        tail products, idle-CTA fill, same-class P5C bodies for short wide and long-K partition products); None
        leaves lowering unchanged.
        accumulation_rules='castview' opts into ops.accumulation_rules' gradient-traffic rules (exact
        widening casts read per view are evaluated per view instead of stored); None leaves lowering unchanged.
        memory_rules='sink' opts into ops.memory_rules' activation lifetime rules (bodies move later, never
        earlier, when that lowers live arena bytes; values unchanged); None leaves lowering unchanged.
        exact_bf16_products='tensor' runs FP32 products of exact BF16 -> FP32 operand widenings (e.g. the FP32-result
        split products of an opt-in training precision policy) on the masked TMA wgmma body from the BF16 sources with
        FP32 accumulation and an unrounded FP32 result, outside the compose planner too; None leaves lowering unchanged.
        Omitted execution (or 'auto') chooses a compatible common or retained
        implementation. 'legacy' and 'compose' remain diagnostic overrides;
        compose_selection='common' disables retained-schedule selection.
        precision_policy='practical' permits valid BF16 execution reassociation while
        preserving captured dtypes. bf16_weight_chunk_k optionally controls BF16
        partials in eligible weight-like products, accumulated in FP32.
        square_sum_association='balanced' additionally opts into balanced FP32
        scalar square-sum reductions under practical permission.
        """
        if type(static_shared_reserve_bytes) is not int or static_shared_reserve_bytes < 0:
            raise ValueError("static_shared_reserve_bytes must be a nonnegative integer")
        return prepare_module(self, directory, cache_dir=cache_dir,
                              arch=arch, retain_all=retain_all, policy=policy,
                              precision_policy=precision_policy, compact_codegen=compact_codegen,
                              execution=execution, compose_selection=compose_selection, tuning=tuning,
                              gradient_outputs=gradient_outputs, fast_build=fast_build,
                              fast_device_compile=fast_device_compile,
                              native_code_only=native_code_only,
                              disable_expensive_optimizations=disable_expensive_optimizations,
                              ptxas_optimization_level=ptxas_optimization_level,
                              recurrent_gradients=recurrent_gradients, explain=explain,
                              bf16_weight_chunk_k=bf16_weight_chunk_k,
                              square_sum_association=square_sum_association, device_units=device_units,
                              device_unit_inline=device_unit_inline, device_unit_cache=device_unit_cache,
                              device_unit_jobs=device_unit_jobs, device_unit_split=device_unit_split,
                              device_unit_shared_pool=device_unit_shared_pool,
                              **({'static_shared_reserve_bytes': static_shared_reserve_bytes}
                                 if static_shared_reserve_bytes != 0 else {}),
                              **({'update_schedule': update_schedule} if update_schedule is not None else {}),
                              **({'segmented_products': segmented_products} if segmented_products is not None else {}),
                              **({'norm_fusion': norm_fusion} if norm_fusion is not None else {}),
                              **({'attention_rules': attention_rules} if attention_rules is not None else {}),
                              **({'matrix_schedule': matrix_schedule} if matrix_schedule is not None else {}),
                              **({'accumulation_rules': accumulation_rules} if accumulation_rules is not None else {}),
                              **({'memory_rules': memory_rules} if memory_rules is not None else {}),
                              **({'matrix_bodies': matrix_bodies} if matrix_bodies is not None else {}),
                              **({'exact_bf16_products': exact_bf16_products} if exact_bf16_products is not None
                                 else {}))


def capture_module(model, loss_fn, inputs, targets=(), *, optimizer=None,
                   optimizer_state=None, state=None, precision=None, dropout_seed=None, dynamic_batch=None,
                   gradient_transform=None, backward_autocast=None, source_locations=False):
    """Capture ``loss_fn(model(*inputs), *targets)`` and an optimizer update.

    ``inputs`` and ``targets`` are tuples of explicit tensors. Tuple/dict model
    outputs are allowed when the loss handles them. Registered buffers are bound
    as explicit read-only inputs, and identities tied within each group are kept.
    Parameters with requires_grad=False at capture are frozen (ops/frozen.py):
    private read-only snapshots bound like buffers, with no gradient, optimizer
    state, decay or update. Changing trainability later requires recapture.
    Buffers sharing storage with mutable parameters or explicit state are refused.
    Python loops/branches specialize; runtime control uses the separate region
    APIs. Caller-visible mutation and implicit PyTorch randomness are refused,
    except exact ``nn.Dropout`` modules: release step 08 (ops/module_dropout.py)
    translates them before AD into the explicit Philox dropout operation and
    appends one int64[4] model state ``dropout_rng`` seeded by ``dropout_seed``
    (default ``torch.initial_seed()``). Evaluation-mode modules are identity.

    With an explicit ``state`` mapping, forward instead receives ``state=state``
    and returns ``(prediction, new_state)``. Registered buffers remain read-only;
    mutable model or random state uses this explicit state contract.

    precision selects an ops.precision forward arithmetic policy before AD.
    None preserves the program as written; policies are not CUDA autocast.
    gradient_transform(grads, canonical_names) optionally transforms the full
    gradient tuple once after AD, before both public output and optimizer use.
    backward_autocast="bf16" enables device-specific BF16 autocast only for AD;
    the forward, gradient transform and optimizer run with autocast disabled.
    None preserves existing behavior. CPU capture uses CPU autocast dispatch.

    LATCH-STEP-11: a forward/loss that calls ``torch.cond(pred, true_fn,
    false_fn, operands)`` with ``pred`` one of the explicit inputs/targets (one
    torch.bool element) captures a uniform device conditional. ``stage`` binds
    the predicate like any other input; changing it selects the other complete
    update in the same kernel without recompilation.
    DynamicBatch(maximum=N) admits leading batches 1..N for contiguous rank-two
    input/target examples of capacity N. CPU preparation audits each size and
    emits one capacity program whose execution extent is a device scalar.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError('capture_module expects an nn.Module')
    if not isinstance(inputs, (tuple, list)) or not isinstance(targets, (tuple, list)):
        raise TypeError('inputs and targets must be tuples/lists of tensors')
    inputs, targets = tuple(inputs), tuple(targets)
    if dynamic_batch is not None and backward_autocast is not None:
        raise CaptureError('BACKWARD_AUTOCAST', 'backward autocast requires static batches')
    if dynamic_batch is not None and gradient_transform is not None:
        raise CaptureError('GRADIENT_TRANSFORM', 'gradient transforms require static batches')
    if dynamic_batch is not None:
        # Preserve the bounded step-19 qualification. Explicit Philox state is
        # supported; module-dropout translation and frozen-weight combinations
        # require their own active-extent qualification before admission.
        from .module_dropout import plan_dropout
        dynamic_dropout = plan_dropout(model, dropout_seed)
        if dynamic_dropout is not None and dynamic_dropout.random:
            raise CaptureError('DYNAMIC_BATCH', 'dynamic batch supports explicit Philox dropout; '
                               'active nn.Dropout is outside its qualified scope')
        if any(not value.requires_grad for value in model.parameters()):
            raise CaptureError('DYNAMIC_BATCH', 'dynamic batch with frozen parameters is outside '
                               'its qualified scope')
        def capture_at(batch_inputs, batch_targets):
            return capture_module(model, loss_fn, batch_inputs, batch_targets,
                                  optimizer=optimizer, optimizer_state=optimizer_state,
                                  state=state, precision=precision, dropout_seed=dropout_seed)
        return capture_dynamic_module(capture_at, inputs, targets, dynamic_batch)
    from .frozen import check_optimizer_names, partition_parameters
    all_params = dict(model.named_parameters(remove_duplicate=False))
    if not all_params:
        raise CaptureError('MODULE_PARAMETERS', 'the module registers no parameters to train')
    partition = partition_parameters(all_params, mutable=(*(optimizer_state or {}).values(),
                                                          *(state or {}).values()))
    check_optimizer_names(optimizer or SGDConfig(), partition, optimizer_state)
    params = partition.trainable
    frozen_names = partition.frozen_names
    mutable = {*(_storage_key(value) for value in all_params.values()),
               *(_storage_key(value) for value in (optimizer_state or {}).values()),
               *(_storage_key(value) for value in (state or {}).values())} - {None}
    buffers, buffer_aliases, identities = {}, {}, {}
    for name, value in model.named_buffers(remove_duplicate=False):
        if _storage_key(value) in mutable:
            raise CaptureError('MODULE_BUFFER_ALIAS',
                               f'buffer {name} shares mutable parameter/optimizer/model-state storage; '
                               'represent the value through parameters or explicit state')
        canonical = identities.setdefault(id(value), name)
        buffer_aliases[name] = canonical
        if canonical not in buffers:
            # Freeze a private snapshot, rather than silently reading future
            # mutations of the user's Python module during native execution.
            buffers[canonical] = value.detach().clone()
    buffer_names = tuple(buffers)
    ni, nt, nb = len(inputs), len(targets), len(buffer_names)
    from contextlib import nullcontext
    from .module_dropout import STATE_NAME, bind, plan_dropout
    plan = plan_dropout(model, dropout_seed)
    random = plan is not None and plan.random
    capture_state = state
    if random:
        if state is not None and STATE_NAME in state:
            raise CaptureError('DROPOUT_STATE', f'explicit state name {STATE_NAME!r} is reserved for nn.Dropout')
        device = next(iter(params.values())).device
        capture_state = {**(state or {}), STATE_NAME: plan.initial_state(device)}

    def run(params, explicit_state, arguments):
        module_inputs = tuple(arguments[:ni])
        target_inputs = tuple(arguments[ni:ni + nt])
        canonical_buffers = dict(zip(buffer_names, arguments[ni + nt:ni + nt + nb]))
        bound_buffers = {alias: canonical_buffers[canonical]
                         for alias, canonical in buffer_aliases.items()}
        bound_frozen = partition.bound(dict(zip(frozen_names, arguments[ni + nt + nb:])))
        rng, user_state = None, explicit_state
        if random:
            rng = explicit_state[STATE_NAME]
            user_state = {k: v for k, v in explicit_state.items() if k != STATE_NAME}
        kwargs = {'state': user_state} if state is not None else None
        with bind(model, plan, rng) if plan is not None else nullcontext() as carrier:
            result = functional_call(model, (params, bound_frozen, bound_buffers), module_inputs,
                                     kwargs=kwargs, tie_weights=True, strict=True)
        if state is not None:
            if not isinstance(result, tuple) or len(result) != 2:
                raise CaptureError('MODULE_STATE', 'stateful module must return (prediction, new_state)')
            prediction, next_state = result
        else:
            prediction, next_state = result, {}
        loss = loss_fn(prediction, *target_inputs)
        if random:
            return loss, {**next_state, STATE_NAME: carrier.state}
        return (loss, next_state) if state is not None else loss

    if capture_state is None:
        def loss(params, *arguments):
            return run(params, None, arguments)
    else:
        def loss(params, explicit_state, *arguments):
            return run(params, explicit_state, arguments)
    flat_inputs = (*inputs, *targets, *buffers.values(), *partition.frozen.values())
    try:
        from .capture import _source_location_scope, module_path_recording
        with _source_location_scope(source_locations), module_path_recording(model, source_locations):
            captured = capture_training(loss, params, None, flat_inputs, optimizer or SGDConfig(),
                                        optimizer_state=optimizer_state, state=capture_state, precision=precision,
                                        conditional='refuse' if random else 'auto', _cond_inputs=ni + nt,
                                        source_locations=source_locations,
                                        gradient_transform=gradient_transform, backward_autocast=backward_autocast)
    except CaptureError as error:
        # Reject at first detection, before branch specialization can expose a
        # less specific constant/region ABI error for the implicit RNG carrier.
        if random and error.code == 'COND_SCOPE':
            raise CaptureError('COND_DROPOUT', 'ordinary nn.Dropout inside torch.cond is outside '
                               'the qualified conditional capture scope') from None
        raise
    frontend = getattr(captured, 'frontend', None)
    if frontend is not None and random:
        raise CaptureError('COND_DROPOUT', 'ordinary nn.Dropout inside torch.cond is outside '
                           'the qualified conditional capture scope')
    if frontend is not None and frontend['predicate_input'] >= ni + nt:
        raise CaptureError('COND_PREDICATE', 'a registered buffer is a frozen capture snapshot; pass the torch.cond '
                           'predicate as an explicit forward input or target')
    # Input roles are the contract; do not assume optimizer state has one tensor
    # per parameter (AdamW has moments and independent counters).
    positions = [index for index, name in enumerate(captured.ir.inputs)
                 if captured.ir.values[name].role == 'input']
    runtime_lr = captured.ir.metadata.get('optimizer_lr_input')
    if runtime_lr is not None:
        index = runtime_lr['index']
        if index not in positions:
            raise CaptureError('MODULE_ABI', 'runtime learning-rate binding is not a data input')
        positions.remove(index)
    if frontend is not None:
        # The region ABI appends the predicate; restore its user position.
        positions.insert(frontend['predicate_input'], frontend['region_predicate_index'])
    if len(positions) != len(flat_inputs):
        raise CaptureError('MODULE_ABI', 'captured module data/buffer signature changed')
    indices = {'inputs': positions[:ni], 'targets': positions[ni:ni + nt],
               'buffers': dict(zip(buffer_names, positions[ni + nt:ni + nt + nb]))}
    if frozen_names:
        indices['frozen'] = dict(zip(frozen_names, positions[ni + nt + nb:]))
    if runtime_lr is not None:
        indices['optimizer_lr'] = runtime_lr['index']
    captured.ir.metadata['module_api'] = {
        'version': 1, 'input_indices': indices, 'buffer_aliases': buffer_aliases,
        'buffer_policy': 'private capture snapshot; read-only explicit bindings',
        'stateful_forward': state is not None,
        'loss_signature': 'loss_fn(prediction, *targets)',
        'python_control': ('static specialization; one torch.cond predicate input selects a uniform device region'
                           if frontend is not None else
                           'static specialization; runtime control requires explicit regions')}
    if frozen_names:
        captured.ir.metadata['module_api'].update(
            frozen_aliases=dict(partition.frozen_aliases), frozen_policy=partition.describe(optimizer or SGDConfig()))
    if plan is not None:
        captured.ir.metadata['module_api']['dropout'] = plan.metadata()
    return CapturedModule(captured, indices, buffer_aliases)


def prepare_module(captured, directory, *, cache_dir, arch='sm_90', retain_all=False, policy='auto',
                   precision_policy=None, compact_codegen='off', execution=None, compose_selection='auto',
                   tuning=None, matrix_choices_scope=None, gradient_outputs='returned', fast_build=False,
                   fast_device_compile=False,
                   native_code_only=False, disable_expensive_optimizations=False, ptxas_optimization_level=None,
                   recurrent_gradients='inline', explain=False, bf16_weight_chunk_k=None,
                   square_sum_association=None, device_units=None, device_unit_inline='wgmma',
                   device_unit_cache=None, device_unit_jobs=None, device_unit_split=None,
                   device_unit_shared_pool=False, static_shared_reserve_bytes=0, update_schedule=None,
                   segmented_products=None, norm_fusion=None, attention_rules=None, matrix_schedule=None,
                   accumulation_rules=None, memory_rules=None, matrix_bodies=None, exact_bf16_products=None):
    """Persist source, metadata and initial bindings with exact library identity.

    The receipt binds the recorded library path and toolchain. This is not a
    claim that native caches can move between arbitrary hosts or CUDA versions.
    ``tuning`` routes to ``ops.matrix_tuning.prepare_tuned``; ``matrix_choices_scope``
    is that module's internal (choices, inventory log[, default policy]) tuple for one variant.
    """
    from contextlib import nullcontext
    from .lower import lower
    from .runtime import build
    from .practical_bf16 import validate as validate_execution_precision
    from .square_sum_association import validate_option as validate_square_sum_association
    validate_execution_precision(precision_policy, bf16_weight_chunk_k)
    validate_square_sum_association(precision_policy, square_sum_association)
    from . import update_schedule as update_schedules
    update_schedules.validate(update_schedule)
    from . import ragged_experts_lower
    ragged_experts_lower.validate_products(segmented_products)
    from .lower import exact_bf16_products_scope, validate_exact_bf16_products
    validate_exact_bf16_products(exact_bf16_products)
    if exact_bf16_products is not None and tuning is not None:
        raise ValueError("exact_bf16_products is not yet supported with matrix tuning")
    if segmented_products is not None and tuning is not None:
        raise ValueError("segmented_products is not yet supported with matrix tuning")
    if update_schedule is not None and tuning is not None:
        raise ValueError("update_schedule is not yet supported with matrix tuning")
    from . import norm_fusion as norm_fusions
    norm_fusions.validate(norm_fusion)
    if norm_fusion is not None and tuning is not None:
        raise ValueError("norm_fusion is not yet supported with matrix tuning")
    from . import accumulation_rules as accumulation_rule_sets
    accumulation_rule_sets.validate(accumulation_rules)
    if accumulation_rules is not None and tuning is not None:
        raise ValueError("accumulation_rules is not yet supported with matrix tuning")
    from . import memory_rules as memory_rule_sets
    memory_rule_sets.validate(memory_rules)
    if memory_rules is not None and tuning is not None:
        raise ValueError("memory_rules is not yet supported with matrix tuning")
    from . import attention_rules as attention_rule_sets
    attention_rule_sets.validate(attention_rules)
    if attention_rules is not None and tuning is not None:
        raise ValueError("attention_rules is not yet supported with matrix tuning")
    from . import matrix_schedule as matrix_schedules
    matrix_schedules.validate(matrix_schedule)
    if matrix_schedule is not None and tuning is not None:
        raise ValueError("matrix_schedule is not yet supported with matrix tuning")
    from . import matrix_bodies as matrix_body_sets
    matrix_body_sets.validate(matrix_bodies)
    if matrix_bodies is not None and tuning is not None:
        raise ValueError("matrix_bodies is not yet supported with matrix tuning")
    if type(static_shared_reserve_bytes) is not int or static_shared_reserve_bytes < 0:
        raise ValueError("static_shared_reserve_bytes must be a nonnegative integer")
    if static_shared_reserve_bytes and tuning is not None:
        raise ValueError("static shared reserve is not yet supported with matrix tuning")
    if type(fast_device_compile) is not bool:
        raise ValueError("fast_device_compile must be a boolean")
    if type(native_code_only) is not bool:
        raise ValueError("native_code_only must be a boolean")
    if type(disable_expensive_optimizations) is not bool:
        raise ValueError("disable_expensive_optimizations must be a boolean")
    if ptxas_optimization_level is not None and (
            type(ptxas_optimization_level) is not int or ptxas_optimization_level not in range(4)):
        raise ValueError("ptxas_optimization_level must be None or an integer from 0 through 3")
    if (compact_codegen == 'elementwise_full_sum_resource_pure_runs_attention_items' and
            ptxas_optimization_level != 0):
        raise ValueError("attention-item outlining requires explicit ptxas_optimization_level=0; only this level is qualified")
    if ptxas_optimization_level is not None and (fast_build or fast_device_compile):
        raise ValueError("explicit ptxas optimization does not combine with host-shim or fast-device builds")
    if native_code_only and fast_build:
        raise ValueError("native_code_only does not combine with fast_build host-shim builds")
    if disable_expensive_optimizations and fast_build:
        raise ValueError("disable_expensive_optimizations does not combine with fast_build host-shim builds")
    if device_units is not None and (fast_build or fast_device_compile or tuning is not None):
        raise ValueError("device_units does not combine with fast_build, fast_device_compile or matrix tuning")
    if tuning is not None:
        if ptxas_optimization_level is not None:
            raise ValueError("explicit ptxas optimization is not yet supported with matrix tuning")
        if disable_expensive_optimizations:
            raise ValueError("disable_expensive_optimizations is not yet supported with matrix tuning")
        if native_code_only:
            raise ValueError("native_code_only is not yet supported with matrix tuning")
        if fast_device_compile:
            raise ValueError("fast_device_compile is not yet supported with matrix tuning")
        if square_sum_association is not None:
            raise ValueError("square_sum_association is not yet supported with matrix tuning")
        if bf16_weight_chunk_k is not None:
            raise ValueError("bf16_weight_chunk_k is not yet supported with matrix tuning")
        from .matrix_tuning import prepare_tuned
        return prepare_tuned(captured, directory, cache_dir=cache_dir, arch=arch, retain_all=retain_all,
                             policy=policy, precision_policy=precision_policy, compact_codegen=compact_codegen,
                             execution=execution, compose_selection=compose_selection, tuning=tuning,
                             gradient_outputs=gradient_outputs, fast_build=fast_build,
                             recurrent_gradients=recurrent_gradients)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if matrix_choices_scope is not None:
        from .matrix_adapters.provider import matrix_choices
        scope = matrix_choices(matrix_choices_scope[0], log=matrix_choices_scope[1],
                               policy=matrix_choices_scope[2] if len(matrix_choices_scope) > 2 else None)
    else:
        scope = nullcontext()
    diagnostics = None
    # Opaque BF16 SDPA has a paired primal/VJP tape. The distributed entry point
    # exposes that tape before partitioning; an ordinary single-module build
    # needs the same common operation rewrite without creating extra regions.
    # Keep capture unchanged so callers that partition later can select their
    # own admitted attention schedule.
    native_ir = captured.ir
    from . import bf16_sdpa
    if (not getattr(native_ir, 'metadata', {}).get('composed_attention_tape')
            and any(op.target == bf16_sdpa.FORWARD for op in getattr(native_ir, 'operations', ()))):
        from .compose_attention import rewrite_tape
        with attention_rule_sets.scope(attention_rules):
            native_ir, _ = rewrite_tape(native_ir, version=7)
    # LATCH-SELECTIVE-10 dcast (ops.accumulation_rules): persistent derived BF16 copies of cast parameters extend
    # the captured state ABI (one appended state binding per copy) before lowering.
    example_args, derived_state = captured.example_args, []
    if 'dcast' in accumulation_rule_sets.rules_of(accumulation_rules):
        from . import derived_casts
        native_ir, example_args, derived_state = derived_casts.apply(native_ir, example_args)
    if explain:
        from .explain import collect as _explain_collect
        explain_scope = _explain_collect()
    else:
        explain_scope = nullcontext()
    with scope, explain_scope as diagnostics, update_schedules.scope(update_schedule), \
            ragged_experts_lower.products_scope(segmented_products), \
            norm_fusions.scope(norm_fusion), \
            attention_rule_sets.scope(attention_rules), \
            matrix_schedules.scope(matrix_schedule), \
            matrix_body_sets.scope(matrix_bodies), \
            accumulation_rule_sets.scope(accumulation_rules), \
            memory_rule_sets.scope(memory_rules), \
            exact_bf16_products_scope(exact_bf16_products):
        program = lower(native_ir, retain_all=retain_all, policy=policy, target_arch=arch,
                        precision_policy=precision_policy, compact_codegen=compact_codegen,
                        execution=execution, compose_selection=compose_selection,
                        recurrent_gradients=recurrent_gradients,
                        **({"bf16_weight_chunk_k": bf16_weight_chunk_k} if bf16_weight_chunk_k is not None else {}),
                        **({"square_sum_association": square_sum_association} if square_sum_association is not None else {}),
                        **({'static_shared_reserve_bytes': static_shared_reserve_bytes}
                           if static_shared_reserve_bytes != 0 else {}),
                        **({} if gradient_outputs == 'returned' else {'gradient_outputs': gradient_outputs}))
    if derived_state:
        program.manifest['derived_state'] = derived_state
    program.write(directory)
    torch.save({'args': example_args, 'constants': captured.constants},
               directory / 'bindings.pt')
    started_build = time.perf_counter()
    library = build(program, cache_dir=cache_dir, arch=arch,
                    **({'fast_device_compile': True} if fast_device_compile else {}),
                    **({'native_code_only': True} if native_code_only else {}),
                    **({'disable_expensive_optimizations': True} if disable_expensive_optimizations else {}),
                    **({'ptxas_optimization_level': ptxas_optimization_level} if ptxas_optimization_level is not None else {}),
                    **({'step17': 'qualified'} if fast_build else {}),
                    **({'device_units': device_units, 'device_unit_inline': device_unit_inline,
                        'device_unit_cache': device_unit_cache, 'device_unit_jobs': device_unit_jobs,
                        'device_unit_split': device_unit_split, 'device_unit_shared_pool': device_unit_shared_pool}
                       if device_units is not None else {}))
    build_seconds = time.perf_counter() - started_build
    files = {path.name: _hash(path) for path in sorted(directory.iterdir()) if path.is_file()}
    receipt = {'schema': 'latch-ops-module-artifact-v1', 'source_hash': program.cache_key,
               'files': files, 'library': str(library), 'library_sha256': _hash(library),
               'arch': arch, 'torch': torch.__version__, 'build_seconds': build_seconds,
               'execution': program.manifest.get('execution_plan', {}).get('selected', 'legacy'),
               'prepare_seconds': time.perf_counter() - start,
               'module_api': captured.ir.metadata['module_api'],
               'parameter_aliases': captured.parameter_aliases,
               'optimizer': captured.ir.metadata['optimizer'],
               'gradient_outputs': gradient_outputs, 'fast_build': bool(fast_build),
               'recurrent_gradients': recurrent_gradients}
    if fast_device_compile:
        receipt['fast_device_compile'] = True
    if native_code_only:
        receipt['native_code_only'] = True
    if disable_expensive_optimizations:
        receipt['disable_expensive_optimizations'] = True
    if ptxas_optimization_level is not None:
        receipt['ptxas_optimization_level'] = ptxas_optimization_level
    if device_units is not None:
        receipt['device_units'] = dict(units=device_units, inline=device_unit_inline, split=device_unit_split,
                                       shared_pool=bool(device_unit_shared_pool))
    if 'execution_selection' in program.manifest:
        receipt['execution_selection'] = program.manifest['execution_selection']
    receipt['immutable_bindings_sha256'] = _immutable_identity(
        example_args, program.manifest, receipt['module_api'])
    (directory / 'artifact.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    if explain:
        # LATCH-EXPLAIN: side directory; the receipt's file list (regular files of ``directory``) is unchanged.
        from .explain import report_program, write_report
        side = directory / 'explain'
        side.mkdir(exist_ok=True)
        ir_dict = native_ir.to_dict()
        (side / 'ir.json').write_text(json.dumps(ir_dict, default=repr) + '\n')
        (side / 'diagnostics.json').write_text(json.dumps(diagnostics or [], default=repr) + '\n')
        report = report_program(program, ir=ir_dict, diagnostics=diagnostics or [],
                                subject={'kind': 'artifact', 'path': str(directory)})
        write_report(report, side)
        receipt['explain'] = {'directory': str(side), 'schema': report['schema']}
    return receipt


class ModuleUpdate:
    """Bound update with explicit minibatch staging and separate persistent state."""
    def __init__(self, runtime, metadata):
        self.runtime, self.metadata = runtime, metadata

    def __getattr__(self, name):
        if name == 'runtime':
            raise AttributeError(name)
        return getattr(self.runtime, name)

    @staticmethod
    def allocate_shared_arena(*directories, device='cuda'):
        """Allocate temporary storage for sequentially executed prepared programs.

        Pass the returned tensor as ``arena=`` to each ``load``. Their persistent
        training state must be shared separately with ``bindings=``. Programs
        using this arena must never execute concurrently, including on different
        CUDA streams. Loading still verifies each artifact and its native library.
        """
        from .lower import GeneratedProgram
        from .runtime import _arena_allocation_words
        if not directories:
            raise ValueError('at least one prepared program is required')
        manifests = [GeneratedProgram.load(directory).manifest for directory in directories]
        for manifest in manifests:
            if manifest.get('persistent_counter_words') or any(
                    effect.get('storage') == 'runtime_counter_arena'
                    for phase in manifest.get('phases', [])
                    for effect in (phase.get('counter_effects') or [])):
                raise ValueError('programs with launch-persistent counters cannot share an arena')
        words = max(_arena_allocation_words(manifest['arena_words']) for manifest in manifests)
        return torch.empty(words, device=device, dtype=torch.float32)

    @classmethod
    def load(cls, directory, *, device='cuda', blocks=None, bindings=None, arena=None):
        """Verify and load a CPU-prepared artifact; native compilation is disabled.

        ``bindings`` optionally maps flat argument indices to existing device
        tensors bound instead of fresh copies of the saved values. Each must
        have the saved shape, strides and dtype on ``device``. Batch variants
        use this to share one persistent training state between programs.

        ``arena`` may be allocated by ``allocate_shared_arena`` for programs
        executed sequentially. It holds only temporaries, not persistent state.
        """
        from .lower import GeneratedProgram
        from .runtime import Runtime
        directory = Path(directory)
        receipt = json.loads((directory / 'artifact.json').read_text())
        if receipt.get('schema') != 'latch-ops-module-artifact-v1':
            raise ValueError('unsupported module artifact schema')
        for name, expected in receipt['files'].items():
            if Path(name).name != name or _hash(directory / name) != expected:
                raise ValueError(f'artifact identity mismatch: {name}')
        if _hash(receipt['library']) != receipt['library_sha256']:
            raise ValueError('native library identity mismatch')
        program = GeneratedProgram.load(directory)
        if program.cache_key != receipt['source_hash']:
            raise ValueError('native source identity mismatch')
        saved = torch.load(directory / 'bindings.pt', weights_only=True, map_location='cpu')
        # Older v1 artifacts already bind bindings.pt by hash. Derive the new
        # checkpoint identity from those verified bytes without rebuilding.
        immutable_identity = _immutable_identity(saved['args'], program.manifest, receipt['module_api'])
        if receipt.get('immutable_bindings_sha256', immutable_identity) != immutable_identity:
            raise ValueError('immutable buffer/constant identity mismatch')
        bindings = dict(bindings or {})
        if not set(bindings) <= set(range(len(saved['args']))):
            raise ValueError('shared binding index outside the captured signature')
        if bindings:
            device = torch.empty(0, device=device).device   # resolve an implicit device index
        args = []
        for index, value in enumerate(saved['args']):
            if index in bindings:
                bound = bindings[index]
                if (not isinstance(bound, torch.Tensor) or bound.shape != value.shape
                        or bound.stride() != value.stride() or bound.dtype != value.dtype
                        or bound.device != device):
                    raise ValueError(f'shared binding {index} differs from the captured layout/device')
                args.append(bound)
                continue
            # Preserve admitted positive-stride layouts, including dense
            # transposes and slices with gaps. to()/clone() alone can silently
            # make a non-dense slice contiguous and change the captured ABI.
            bound = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
            bound.copy_(value)
            args.append(bound)
        args = tuple(args)
        # ``arena`` (LATCH-MOESUP): an existing FP32 device vector reused as this program's temporary arena when
        # programs never run concurrently; see Runtime.
        runtime = Runtime(program, args, library=receipt['library'], blocks=blocks, arena=arena)
        return cls(runtime, {**receipt['module_api'], 'immutable_bindings_sha256': immutable_identity})

    def stage(self, inputs, targets=()):
        """Copy a minibatch into stable device bindings, separately from step timing."""
        copies = []
        dynamic = self.runtime.manifest.get('dynamic_batch')
        active_batch = None
        for group, supplied in (('inputs', tuple(inputs)), ('targets', tuple(targets))):
            positions = self.metadata['input_indices'][group]
            if len(positions) != len(supplied):
                raise ValueError(f'{group} arity differs from capture')
            for index, source in zip(positions, supplied):
                target = self.runtime.args[index]
                if not isinstance(source, torch.Tensor) or source.dtype != target.dtype:
                    raise ValueError(f'{group} shape/dtype differs from capture')
                if dynamic is None:
                    if source.shape != target.shape:
                        raise ValueError(f'{group} shape/dtype differs from capture')
                    destination = target
                else:
                    if (source.ndim != target.ndim or source.shape[1:] != target.shape[1:]
                            or not 1 <= source.shape[0] <= dynamic['maximum']):
                        raise ValueError(f'{group} batch shape is outside captured capacity')
                    batch = source.shape[0]
                    if active_batch is not None and batch != active_batch:
                        raise ValueError('all staged inputs and targets must have the same active batch')
                    active_batch = batch
                    destination = target[:batch]
                copies.append((destination, source))
        _copy_bindings(copies)
        if active_batch is not None:
            self.runtime.set_active_batch(active_batch)

    def set_learning_rate(self, value):
        """Stage a finite nonnegative runtime AdamW rate before the update.

        Tensor rates are explicit scalar FP32 bindings; the caller guarantees
        their numerical domain, as with other runtime tensor inputs.
        """
        index = self.metadata['input_indices'].get('optimizer_lr')
        if index is None:
            raise ValueError('this program captured a static learning rate')
        target = self.runtime.args[index]
        with torch.no_grad():
            if isinstance(value, torch.Tensor):
                if value.shape != target.shape or value.dtype != target.dtype:
                    raise ValueError('learning-rate shape/dtype differs from capture')
                target.copy_(value)
            else:
                import math
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError('learning rate must be finite and nonnegative')
                target.fill_(value)

    def frozen_parameters(self):
        """Device views of the read-only frozen-parameter bindings, by canonical name."""
        return {name: self.runtime.args[index]
                for name, index in self.metadata['input_indices'].get('frozen', {}).items()}

    def step(self):
        """One complete native training launch; outputs are reused arena views."""
        return self.runtime.step()

    def state_dict(self):
        """Take a synchronized CPU snapshot of parameter/optimizer/model state."""
        result = {'source_hash': self.runtime.program.cache_key,
                  'immutable_bindings_sha256': self.metadata['immutable_bindings_sha256']}
        for kind in ('parameters', 'optimizer_state', 'state'):
            result[kind] = {name: value.detach().cpu().clone()
                            for name, value in getattr(self.runtime, kind).items()}
        if self.runtime.manifest.get('dynamic_batch') is not None:
            result['active_batch'] = self.runtime.active_batch_tensor.detach().cpu().clone()
        return result

    def load_state_dict(self, state):
        """Restore this exact program's persistent bindings without changing pointers."""
        if state.get('source_hash') != self.runtime.program.cache_key:
            raise ValueError('snapshot belongs to a different generated program')
        if state.get('immutable_bindings_sha256') != self.metadata['immutable_bindings_sha256']:
            raise ValueError('snapshot belongs to different frozen buffers/constants')
        dynamic = self.runtime.manifest.get('dynamic_batch')
        batch = None
        if dynamic is not None:
            saved_batch = state.get('active_batch')
            if (not isinstance(saved_batch, torch.Tensor) or saved_batch.dtype != torch.int64
                    or saved_batch.shape != ()):
                raise ValueError('dynamic snapshot needs a scalar int64 active_batch')
            batch = int(saved_batch.item())
            if not 1 <= batch <= dynamic['maximum']:
                raise ValueError('restored active batch is outside captured capacity')
        elif 'active_batch' in state:
            raise ValueError('static program cannot restore dynamic batch state')
        _copy_bindings(_restore_copies(self.runtime, state))
        if batch is not None:
            self.runtime.set_active_batch(batch)
