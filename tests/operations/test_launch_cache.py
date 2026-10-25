"""Bounded training-runtime admission contracts; numerical work runs on Modal.

``run`` uses only mocked storage metadata. ``run_gpu`` accepts an already built
training program and its CUDA bindings: no compilation, tracing or model lookup.
"""
from __future__ import annotations

from contextlib import nullcontext
import ctypes
from types import SimpleNamespace
import traceback
from unittest.mock import patch

from megakernel_lab.operation_compiler.launch_cache import admitted_launch_source
from megakernel_lab.operation_compiler.runtime import Runtime


def _refused(fn, text):
    try:
        fn()
    except RuntimeError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f'expected refusal: {text}')


def run():
    results = {}

    def check(name, fn):
        try:
            fn()
            results[name] = {'passed': True}
        except Exception:
            results[name] = {'passed': False, 'traceback': traceback.format_exc()}

    class Tensor:
        shape, dtype = (8,), 'float32'

        def __init__(self, pointer, device):
            self.pointer, self.device = pointer, device

        def data_ptr(self): return self.pointer
        def stride(self): return (1,)
        def storage_offset(self): return 0
        def is_neg(self): return False
        def is_conj(self): return False

    prototype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_int, ctypes.c_int)

    def runtime(requested=True, *, symbol=True, identity_available=True):
        rt = Runtime.__new__(Runtime)
        rt.device = SimpleNamespace(index=0)
        rt._tensor_maps = None
        rt.args = (Tensor(16, rt.device),)
        rt.arena, rt._pointers, rt._ticks = (Tensor(p, rt.device) for p in (32, 48, 64))
        rt.torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda d: nullcontext()))
        rt._lib = SimpleNamespace()
        if symbol:
            rt._lib.latch_launch_admitted = prototype(lambda *args: 0)
        context = [1729]
        if identity_available:
            with patch('megakernel_lab.operation_compiler.launch_cache.ContextIdentity', return_value=lambda: context[0]):
                rt._configure_launch_cache(requested)
        else:
            with patch('megakernel_lab.operation_compiler.launch_cache.ContextIdentity', side_effect=OSError('missing driver symbol')):
                rt._configure_launch_cache(requested)
        return rt, context

    def enabled_and_context():
        rt, context = runtime()
        assert rt.launch_cache_policy['enabled'] and rt.launch_cache_policy['context_id'] == 1729
        rt._validate_launch_binding()
        context[0] += 1
        _refused(rt._validate_launch_binding, 'CUDA context changed')

    check('unique_context_reuse_and_refusal', enabled_and_context)

    def bindings():
        rt, _ = runtime()
        rt._validate_launch_binding()
        rt.args[0].pointer += 16
        _refused(rt._validate_launch_binding, 'storage or layout changed')
        rt, _ = runtime()
        rt._tensor_maps = None
        rt.args = (Tensor(16, rt.device),)  # Same address is insufficient after rebinding.
        _refused(rt._validate_launch_binding, 'storage or layout changed')
        rt, _ = runtime()
        rt.arena.pointer += 16
        _refused(rt._validate_launch_binding, 'storage or layout changed')

    check('argument_and_arena_rebinding_refused', bindings)

    def maps():
        rt, _ = runtime(requested=False)
        rt._tensor_maps = Tensor(80, rt.device)
        rt._configure_launch_cache(False)
        rt._validate_launch_binding()
        rt._tensor_maps.pointer += 16
        _refused(rt._validate_launch_binding, 'storage or layout changed')

    check('tensor_map_storage_rebinding_refused', maps)

    def device_and_function():
        rt, _ = runtime()
        rt.device = SimpleNamespace(index=1)
        _refused(rt._validate_launch_binding, 'device changed')
        rt, _ = runtime()
        rt._lib.latch_launch_admitted = prototype(lambda *args: 0)
        _refused(rt._validate_launch_binding, 'launch function changed')

    check('device_and_function_rebinding_refused', device_and_function)

    def fallbacks():
        for settings in ({'requested': False}, {'symbol': False}, {'identity_available': False}):
            rt, _ = runtime(**settings)
            assert not rt.launch_cache_policy['enabled'], settings
            rt._validate_launch_binding()
            rt.args[0].pointer += 16
            _refused(rt._validate_launch_binding, 'storage or layout changed')

    check('diagnostic_and_missing_symbol_fallback', fallbacks)

    def host_source():
        source = admitted_launch_source(17408)
        assert 'static ' not in source and 'thread_local' not in source
        assert 'latch_query(' not in source and 'cudaGetDeviceProperties' not in source
        assert 'device != admitted_device' in source and 'blocks > admitted_max_blocks' in source
        assert source.count('cudaLaunchCooperativeKernel') == 1

    check('stateless_host_source', host_source)
    return {'passed': all(row['passed'] for row in results.values()), 'tests': results}


def run_gpu(program, args, *, library, blocks=None):
    """Qualify cache equivalence/invalidation on one already built training case.

    A temporary empty driver context proves context-ID invalidation without
    resetting PyTorch's context or touching bound storage from another context.
    The owner must call this outside any active CUDA graph capture.
    """
    import torch

    def bindings():
        result = []
        for value in args:
            copy = torch.empty_strided(value.shape, value.stride(), device=value.device, dtype=value.dtype)
            copy.copy_(value)
            result.append(copy)
        return tuple(result)

    cached = Runtime(program, bindings(), library=library, blocks=blocks, launch_cache=True)
    queried = Runtime(program, bindings(), library=library, blocks=blocks, launch_cache=False)
    assert cached.launch_cache_policy['enabled'], cached.launch_cache_policy
    assert not queried.launch_cache_policy['enabled']
    checks = 0

    def equal():
        nonlocal checks
        for a, b in zip((*cached.outputs, *cached.args), (*queried.outputs, *queried.args)):
            torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
            checks += 1

    for _ in range(3):
        cached.step()
        queried.step()
        equal()
    # Same storage permits ordinary staging and one captured launch replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cached.step()
    for rt in (cached, queried):
        for dst, src in zip(rt.args, args):
            dst.copy_(src)
    for _ in range(2):
        graph.replay()
        queried.step()
        equal()
    del graph
    torch.cuda.synchronize(cached.device)

    driver = cached._context_identity.library
    get_current = driver.cuCtxGetCurrent
    get_current.argtypes, get_current.restype = [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int
    create = driver.cuCtxCreate_v2
    create.argtypes, create.restype = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int], ctypes.c_int
    set_current = driver.cuCtxSetCurrent
    set_current.argtypes, set_current.restype = [ctypes.c_void_p], ctypes.c_int
    destroy = driver.cuCtxDestroy_v2
    destroy.argtypes, destroy.restype = [ctypes.c_void_p], ctypes.c_int
    original, temporary = ctypes.c_void_p(), ctypes.c_void_p()
    with torch.cuda.device(cached.device):
        assert get_current(ctypes.byref(original)) == 0
        assert create(ctypes.byref(temporary), 0, cached.device.index) == 0
        try:
            new_context = cached._context_identity()
            assert new_context != cached._admitted_context
            _refused(cached._validate_launch_binding, 'CUDA context changed')
        finally:
            restored = set_current(original)
            released = destroy(temporary)
            assert restored == 0 and released == 0, (restored, released)
        assert cached._context_identity() == cached._admitted_context
        cached._validate_launch_binding()
        stream = torch.cuda.current_stream(cached.device).cuda_stream
        bad_device = cached._fast_launch(cached._pointers.data_ptr(), cached.arena.data_ptr(), None,
                                         cached.blocks, stream, cached.max_blocks, cached.device.index + 1)
        assert bad_device != 0, 'native device guard failed'
    before = cached.launch_count
    replacement = torch.empty_like(cached.args[0])
    with torch.no_grad():
        cached.args[0].set_(replacement)
    _refused(cached.step, 'storage or layout changed')
    assert cached.launch_count == before
    return {'passed': True, 'exact_tensor_comparisons': checks, 'direct_steps': 3, 'graph_replays': 2,
            'cached': cached.launch_cache_policy, 'queried': queried.launch_cache_policy,
            'context_changed_and_restored': True, 'native_device_guard': True,
            'rebound_storage_refused_before_launch': True, 'no_device_reset': True,
            'source_hash': program.cache_key}
