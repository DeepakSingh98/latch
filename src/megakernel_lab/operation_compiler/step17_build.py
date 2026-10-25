"""STEP17: build the native kernel without compiling per-program CUDA host code.

The anchor build (runtime.build) compiles every generated translation unit with
``nvcc --shared``. About 1.7-2.0 s of each build is the host path (host
preprocessing, cudafe++ and the host C++ compile of CUDA runtime headers), the
same for every program, in front of the device path (cicc, ptxas).

When the generated host code is *exactly* the anchor's own template output
(``lower._translation_unit`` + ``launch_cache.admitted_launch_source``, checked by
regenerating that text from the parameters it contains), this helper instead:

1. runs the identical device path once: ``nvcc <BUILD_FLAGS minus --shared> -cubin``;
2. compiles a ~200-line C shim against the CUDA driver API exporting the same
   ``latch_query/latch_launch/latch_error/latch_launch_admitted`` C ABI with the
   same parameters (shared bytes, cluster, grid bound), embedding the cubin.

The result is a self-contained ``program.so`` that the unchanged Runtime loads with
ctypes. Device code is produced by the same cicc/ptxas invocation and flags; the
build receipt records the SASS-level comparison inputs. Any unrecognized host
code (constant tables, cluster-2 P5C launcher, diagnostic exports) is refused and
the caller keeps the anchor nvcc path. Opt-in through ``runtime.build(...,
step17=True)`` or ``LATCH_STEP17_BUILD=on``; the cache key is disjoint from the
anchor key, so neither path can return the other's binary.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

SHIM_VERSION = 'step17-driver-shim-v2'
HOST_NAMES = {'latch_query', 'latch_launch', 'latch_error', 'latch_launch_admitted'}
_EXPORT = re.compile(r'extern "C" (?:int|const char\*) (\w+)\(')


class Unsupported(Exception):
    """The program's host code is not the recognized anchor template; use the anchor build."""


def enabled(requested=None):
    if requested is not None:
        return bool(requested)
    return os.environ.get('LATCH_STEP17_BUILD', '').lower() in {'1', 'on', 'true', 'yes'}


def host_parameters(source):
    """Return (shared_bytes, cluster, max_grid_blocks) iff the host tail is the anchor template verbatim."""
    from .lower import _translation_unit
    if source.count('extern "C" __global__') != 1:
        raise Unsupported('expected exactly one kernel entry')
    exports = set(_EXPORT.findall(source))
    if exports != HOST_NAMES:
        raise Unsupported(f'host exports {sorted(exports)} differ from the anchor template')
    for marker in ('__managed__', '__host__', 'cudaMemcpyToSymbol', 'cudaMemcpyFromSymbol', 'latch_cargs'):
        if marker in source:
            raise Unsupported(f'host-visible marker {marker!r} needs the anchor CUDA runtime build')
    # Vendored headers export diagnostic host functions only under *TRACE* macros (p5c_wg, p8ws, attention_p9).
    if re.search(r'^\s*#\s*define\s+\w*TRACE\w*', source, re.M):
        raise Unsupported('diagnostic trace exports need the anchor CUDA runtime build')
    start = source.index('\nextern "C" int latch_query(')
    tail = source[start:]
    shared = re.search(r'cudaOccupancyMaxActiveBlocksPerMultiprocessor\(&active, latch_ops, 256, (\d+)\)', tail)
    if not shared:
        raise Unsupported('shared-memory size not found in host template')
    shared = int(shared.group(1))
    cluster = re.search(r'clusterDim\.x = (\d+);', tail)
    cluster = int(cluster.group(1)) if cluster else 0
    bound = re.search(r'if \(\*max_blocks > (\d+)\) \*max_blocks = \1;', tail)
    bound = int(bound.group(1)) if bound else None
    expected = _translation_unit('', shared_bytes=shared, cluster=cluster, max_grid_blocks=bound)
    expected_tail = expected[expected.index('\nextern "C" int latch_query('):]
    if tail != expected_tail:
        raise Unsupported('host code differs from the regenerated anchor template')
    return shared, cluster, bound


def _shim_source(shared, cluster, bound):
    """C driver-API equivalent of lower._translation_unit's host functions (same order of checks)."""
    optin = shared > 49152
    nonportable = cluster > 8
    return f'''/* {SHIM_VERSION}: generated; driver-API form of the anchor latch_* host template. */
#include <cuda.h>
#include <pthread.h>
#include <stddef.h>
#include <dlfcn.h>
/* The CUDA driver is resolved on first use (dlopen/dlsym), so the library also loads on machines without a
   driver, as the anchor's statically linked runtime does. Symbol names follow cuda.h's versioned macros. */
#define S17_STR2(x) #x
#define S17_STR(x) S17_STR2(x)
#define S17_DRIVER(X) X(cuCtxGetCurrent) X(cuModuleLoadData) X(cuModuleGetFunction) X(cuModuleUnload) \\
  X(cuCtxGetDevice) X(cuDeviceGetAttribute) X(cuFuncSetAttribute) X(cuOccupancyMaxActiveBlocksPerMultiprocessor) \\
  X(cuOccupancyMaxActiveClusters) X(cuLaunchCooperativeKernel) X(cuLaunchKernelEx) X(cuGetErrorString)
#define S17_DECLARE(f) static __typeof__(&f) s17_##f;
S17_DRIVER(S17_DECLARE)
#define S17(f) (*s17_##f)
static pthread_once_t s17_once = PTHREAD_ONCE_INIT;
static CUresult s17_status = CUDA_ERROR_NOT_INITIALIZED;
static void s17_load(void) {{
  void* library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!library) {{ s17_status = CUDA_ERROR_SHARED_OBJECT_INIT_FAILED; return; }}
#define S17_LOAD(f) s17_##f = (__typeof__(s17_##f))dlsym(library, S17_STR(f)); \\
  if (!s17_##f) {{ s17_status = CUDA_ERROR_NOT_FOUND; return; }}
  S17_DRIVER(S17_LOAD)
  s17_status = CUDA_SUCCESS;
}}
static CUresult s17_driver(void) {{ pthread_once(&s17_once, s17_load); return s17_status; }}
/* Defined by cubin_blob.s (.incbin of program.cubin), assembled after the device compile. */
extern const unsigned char latch_step17_cubin[];
#define SHARED_BYTES {shared}
#define CLUSTER {cluster}
#define OPTIN {int(optin)}
#define NONPORTABLE {int(nonportable)}
#define HAS_BOUND {int(bound is not None)}
#define BOUND {bound if bound is not None else 0}
typedef struct {{ CUcontext context; CUmodule module; CUfunction function; }} latch_entry;
static latch_entry entries[64];
static int entry_count = 0;
static pthread_mutex_t entry_lock = PTHREAD_MUTEX_INITIALIZER;

/* One module per CUDA context, loaded on first use in that context (the runtime API's lazy
   per-context registration). A copied library (private Runtime binding) has its own table. */
static CUresult resolve(CUfunction* function) {{
  CUresult e = s17_driver(); if (e != CUDA_SUCCESS) return e;
  CUcontext context = NULL; e = S17(cuCtxGetCurrent)(&context);
  if (e != CUDA_SUCCESS) return e;
  if (context == NULL) return CUDA_ERROR_INVALID_CONTEXT;
  pthread_mutex_lock(&entry_lock);
  for (int i = 0; i < entry_count; ++i)
    if (entries[i].context == context) {{ *function = entries[i].function; pthread_mutex_unlock(&entry_lock); return CUDA_SUCCESS; }}
  if (entry_count == 64) {{ pthread_mutex_unlock(&entry_lock); return CUDA_ERROR_OUT_OF_MEMORY; }}
  CUmodule module; e = S17(cuModuleLoadData)(&module, latch_step17_cubin);
  if (e == CUDA_SUCCESS) {{
    e = S17(cuModuleGetFunction)(function, module, "latch_ops");
    if (e == CUDA_SUCCESS) {{ entries[entry_count].context = context; entries[entry_count].module = module;
                              entries[entry_count].function = *function; ++entry_count; }}
    else S17(cuModuleUnload)(module);
  }}
  pthread_mutex_unlock(&entry_lock);
  return e;
}}

static CUresult launch(CUfunction function, float** args, float* arena, unsigned long long* ticks, int blocks,
                       void* stream) {{
  void* kernel_args[] = {{&args, &arena, &ticks}};
#if CLUSTER
  blocks -= blocks % CLUSTER;
  if (blocks <= 0) return CUDA_ERROR_INVALID_VALUE;
  CUlaunchAttribute at[2];
  at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION; at[0].value.clusterDim.x = CLUSTER;
  at[0].value.clusterDim.y = 1; at[0].value.clusterDim.z = 1;
  at[1].id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE; at[1].value.cooperative = 1;
  CUlaunchConfig cfg = {{0}};
  cfg.gridDimX = blocks; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = 256; cfg.blockDimY = 1; cfg.blockDimZ = 1;
  cfg.sharedMemBytes = SHARED_BYTES; cfg.hStream = (CUstream)stream; cfg.attrs = at; cfg.numAttrs = 2;
  return S17(cuLaunchKernelEx)(&cfg, function, kernel_args, NULL);
#else
  return S17(cuLaunchCooperativeKernel)(function, blocks, 1, 1, 256, 1, 1, SHARED_BYTES, (CUstream)stream, kernel_args);
#endif
}}

int latch_query(int* max_blocks, int* multiprocessors) {{
  CUfunction function; CUresult e = resolve(&function); if (e != CUDA_SUCCESS) return e;
  CUdevice device; e = S17(cuCtxGetDevice)(&device); if (e != CUDA_SUCCESS) return e;
  int cooperative = 0, sms = 0;
  e = S17(cuDeviceGetAttribute)(&cooperative, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, device); if (e != CUDA_SUCCESS) return e;
  e = S17(cuDeviceGetAttribute)(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device); if (e != CUDA_SUCCESS) return e;
  if (!cooperative) return CUDA_ERROR_NOT_SUPPORTED;
#if OPTIN
  e = S17(cuFuncSetAttribute)(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SHARED_BYTES);
  if (e != CUDA_SUCCESS) return e;
#endif
#if NONPORTABLE
  e = S17(cuFuncSetAttribute)(function, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1);
  if (e != CUDA_SUCCESS) return e;
#endif
  int active; e = S17(cuOccupancyMaxActiveBlocksPerMultiprocessor)(&active, function, 256, SHARED_BYTES);
  if (e != CUDA_SUCCESS) return e;
  *multiprocessors = sms; *max_blocks = active * sms;
#if CLUSTER
  {{
    CUlaunchAttribute at[1];
    at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION; at[0].value.clusterDim.x = CLUSTER;
    at[0].value.clusterDim.y = 1; at[0].value.clusterDim.z = 1;
    CUlaunchConfig cfg = {{0}};
    cfg.gridDimX = CLUSTER; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = 256; cfg.blockDimY = 1; cfg.blockDimZ = 1;
    cfg.sharedMemBytes = SHARED_BYTES; cfg.attrs = at; cfg.numAttrs = 1;
    int clusters = 0; e = S17(cuOccupancyMaxActiveClusters)(&clusters, function, &cfg); if (e != CUDA_SUCCESS) return e;
    if (clusters * CLUSTER < *max_blocks) *max_blocks = clusters * CLUSTER;
  }}
#endif
#if HAS_BOUND
  if (*max_blocks > BOUND) *max_blocks = BOUND;
#endif
  return *max_blocks > 0 ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
}}

int latch_launch(float** args, float* arena, unsigned long long* ticks, int blocks, void* stream) {{
  int max_blocks, sms; int e = latch_query(&max_blocks, &sms); if (e) return e;
  if (blocks <= 0 || blocks > max_blocks) return CUDA_ERROR_INVALID_VALUE;
  CUfunction function; e = resolve(&function); if (e) return e;
  return launch(function, args, arena, ticks, blocks, stream);
}}

const char* latch_error(int error) {{
  const char* text = NULL;
  if (error == CUDA_ERROR_SHARED_OBJECT_INIT_FAILED && s17_driver() != CUDA_SUCCESS) return "CUDA driver library unavailable";
  if (s17_driver() != CUDA_SUCCESS) return "CUDA driver symbols unavailable";
  if (S17(cuGetErrorString)((CUresult)error, &text) != CUDA_SUCCESS || text == NULL) return "unrecognized CUDA driver error code";
  return text;
}}

int latch_launch_admitted(float** args, float* arena, unsigned long long* ticks, int blocks, void* stream,
                          int admitted_max_blocks, int admitted_device) {{
  CUresult e = s17_driver(); if (e != CUDA_SUCCESS) return e;
  CUdevice device; e = S17(cuCtxGetDevice)(&device); if (e != CUDA_SUCCESS) return e;
  if ((int)device != admitted_device) return CUDA_ERROR_INVALID_DEVICE;
  if (blocks <= 0 || blocks > admitted_max_blocks) return CUDA_ERROR_INVALID_VALUE;
  CUfunction function; e = resolve(&function); if (e != CUDA_SUCCESS) return e;
  return launch(function, args, arena, ticks, blocks, stream);
}}
'''


def _cuda_home(nvcc):
    path = shutil.which(nvcc) or nvcc
    return Path(os.path.realpath(path)).parent.parent


def toolchain_identity(nvcc='nvcc', cc='gcc'):
    """Host C compiler and driver header identity, added to the anchor's nvcc key material."""
    home = _cuda_home(nvcc)
    header = home / 'include' / 'cuda.h'
    cc_version = subprocess.run([cc, '--version'], text=True, capture_output=True, check=True).stdout
    return {'shim': SHIM_VERSION, 'cc': cc_version,
            'cuda_h_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
            'shim_template_sha256': hashlib.sha256(_shim_source(1, 0, None).encode()).hexdigest()}


def build(program, directory, *, arch, flags, nvcc='nvcc', cc='gcc', units=1, jobs=None, intervals=None):
    """Write program.so into ``directory`` (already locked/created by the caller). Returns a receipt.

    ``intervals`` (``step17_intervals.identity(...)`` plus an optional ``unit_cache`` directory) selects
    the interval partition: whole publication intervals become device functions, compiled as
    ``intervals['units']`` relocatable units in parallel and device-linked into one cubin with one entry.
    Its failures raise; it never falls back to a different build.
    """
    shared, cluster, bound = host_parameters(program.source)
    directory = Path(directory)
    source = program.write(directory)
    device_flags = [flag for flag in flags if flag != '--shared']
    timings = {}
    cubin = directory / 'program.cubin'
    partition, device, log = None, None, ''
    # The driver shim object does not depend on device code: compile it concurrently (one extra process).
    shim = directory / 'host_shim.c'
    shim.write_text(_shim_source(shared, cluster, bound))
    home = _cuda_home(nvcc)
    shim_command = [cc, '-O2', '-fPIC', '-std=gnu11', f'-I{home / "include"}', '-c', shim.name, '-o', 'host_shim.o']
    shim_started = time.perf_counter()
    shim_proc = subprocess.Popen(shim_command, cwd=directory, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
    if intervals is not None:
        try:
            partition = _build_intervals(program, directory, arch, device_flags, nvcc, intervals, jobs, cubin, timings)
        except BaseException:
            shim_proc.kill()
            shim_proc.wait()
            raise
        log += partition.pop('log', '')
        device = partition['commands']
    elif units > 1:
        partition = _build_partitioned(program, directory, arch, device_flags, nvcc, units, jobs, cubin, timings)
        log += partition.pop('log', '')
        if not partition.get('failed'):
            device = partition['commands']
    if device is None:
        started = time.perf_counter()
        device = [nvcc, *device_flags, f'-arch={arch}', '-cubin', str(source), '-o', str(cubin)]
        proc = subprocess.run(device, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        timings['device_nvcc'] = time.perf_counter() - started
        log += proc.stdout
        if proc.returncode:
            (directory / 'build.log').write_text(log)
            shim_proc.wait()
            raise RuntimeError(f'nvcc -cubin failed ({proc.returncode}); preserved {directory / "build.log"}\n{log[-8000:]}')
    started = time.perf_counter()
    shim_output, _ = shim_proc.communicate()
    timings['host_shim_cc_wait_after_device'] = time.perf_counter() - started
    timings['host_shim_cc_wall'] = time.perf_counter() - shim_started
    log += shim_output
    if shim_proc.returncode:
        (directory / 'build.log').write_text(log)
        raise RuntimeError(f'host shim compile failed ({shim_proc.returncode}); preserved {directory / "build.log"}\n{log[-8000:]}')
    (directory / 'cubin_blob.s').write_text('.section .rodata\n.balign 64\n.globl latch_step17_cubin\n'
                                            '.hidden latch_step17_cubin\nlatch_step17_cubin:\n'
                                            '.incbin "program.cubin"\n.byte 0\n.section .note.GNU-stack,"",@progbits\n')
    temporary = directory / f'program-{os.getpid()}.so'
    host = [cc, '-shared', '-fPIC', 'host_shim.o', 'cubin_blob.s', '-o', temporary.name, '-ldl', '-lpthread']
    started = time.perf_counter()
    proc = subprocess.run(host, cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    timings['embed_link'] = time.perf_counter() - started
    log += proc.stdout
    (directory / 'build.log').write_text(log)
    if proc.returncode:
        raise RuntimeError(f'shim link failed ({proc.returncode}); preserved {directory / "build.log"}\n{log[-8000:]}')
    return dict(mode=SHIM_VERSION, commands=[device, shim_command, host], timings=timings, partition=partition,
                host_parameters=dict(shared_bytes=shared, cluster=cluster, max_grid_blocks=bound),
                cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(), temporary=str(temporary))


PARTITION_VERSION = 'step17-partition-v4'


def partition_identity():
    """Every input that changes the unit sources: pass versions, outline threshold and address-space hint."""
    from .step17_outline import VERSION
    return {'version': PARTITION_VERSION, 'outline': VERSION,
            'outline_min': int(os.environ.get('LATCH_STEP17_OUTLINE_MIN', '300')),
            'assume_global': os.environ.get('LATCH_STEP17_ASSUME_GLOBAL', 'on').lower() in {'1', 'on', 'true', 'yes'},
            'mode': os.environ.get('LATCH_STEP17_PARTITION_MODE', 'chunks'), 'internal_linkage': _internalize(),
            'unit_flags': os.environ.get('LATCH_STEP17_UNIT_FLAGS', '-rdc=true')}


def partition_units(requested=None):
    """Relocatable device units (1 = one device TU), from LATCH_STEP17_UNITS; part of the cache key."""
    value = int(requested if requested is not None else os.environ.get('LATCH_STEP17_UNITS', '1'))
    if not 1 <= value <= 16:
        raise ValueError('LATCH_STEP17_UNITS must be in [1, 16]')
    return value


def _weights(prefix, functions):
    """Approximate device-code volume per outlined block: its text plus the prefix device functions it calls."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(
        r'^__device__ __forceinline__ [\w:<>, *&]+? (\w+)\(', prefix, re.M)]
    sizes = {}
    for (start, name), nxt in zip(starts, starts[1:] + [(len(prefix), None)]):
        sizes[name] = sizes.get(name, 0) + nxt[0] - start
    weights = []
    for _, text in functions:
        called = set(re.findall(r'\b(\w+)\(', text)) & set(sizes)
        templates = len(re.findall(r'::\w+_(?:phase|item)<', text))
        weights.append(len(text) + sum(sizes[c] for c in called) + 3000 * templates)
    return weights


def _secondary_prefix(prefix):
    """Namespace-scope __shared__ definitions become extern declarations; other device variables refuse."""
    out = []
    for line in prefix.split('\n'):
        stripped = line.strip()
        m = re.fullmatch(r'__shared__ ([\w ]+?) (\w+);', stripped)
        if m and line == stripped:
            out.append(f'extern __shared__ {m.group(1)} {m.group(2)};')
            continue
        if line == stripped and re.fullmatch(r'(?:static )?(?:__device__|__constant__|__managed__) [^(]*;', stripped):
            raise Unsupported('namespace-scope device variable needs a single definition: ' + stripped[:80])
        out.append(line)
    return '\n'.join(out)


def partition_sources(source, header_names, units):
    """Return ([(file, text)], stats): unit0 holds the kernel; units 1..k-1 hold balanced outlined blocks."""
    from .step17_outline import Refusal, outline_parts
    try:
        mode = os.environ.get('LATCH_STEP17_PARTITION_MODE', 'chunks')
        if mode not in ('chunks', 'blocks'):
            raise ValueError('LATCH_STEP17_PARTITION_MODE must be chunks or blocks')
        prefix, functions, kernel, stats = outline_parts(source, header_names, min_chars=int(
            os.environ.get('LATCH_STEP17_OUTLINE_MIN', '300')), chunks=units - 1 if mode == 'chunks' else None)
    except Refusal as why:
        raise Unsupported('outline refusal: ' + str(why))
    secondary = _secondary_prefix(prefix)
    renamed = _external_noinline_names(prefix, header_names)
    weights = _weights(prefix, functions)
    bins = [[] for _ in range(max(1, min(units - 1, len(functions))))]
    loads = [0] * len(bins)
    for index in sorted(range(len(functions)), key=lambda i: -weights[i]):
        target = loads.index(min(loads))
        bins[target].append(index)
        loads[target] += weights[index]
    declarations = [text.split('{', 1)[0].strip().replace('static __device__ __noinline__ void', '__device__ void') + ';'
                    for _, text in functions]
    sources = [('unit0.cu', prefix + '// STEP17 relocatable unit 0: kernel entry\n' + '\n'.join(declarations)
                + '\n' + kernel)]
    internal = _internalize()
    system = ''.join(line + '\n' for line in sorted({' '.join(m.split()) for m in re.findall(
        r'^[ \t]*#[ \t]*include[ \t]*<[^>\n]+>', '\n'.join(_header_texts(prefix, header_names)), re.M)})) if internal else ''
    for k, members in enumerate(bins, 1):
        texts = [functions[i][1] for i in sorted(members)]
        if internal:
            sources.append((f'unit{k}.cu', _internalized_unit(secondary, texts, system, k)))
            continue
        body = ''.join(t.replace('static __device__ __noinline__ void', '__device__ __noinline__ void', 1) for t in texts)
        private = ''.join(f'#define {name} {name}_s17u{k}\n' for name in renamed)
        sources.append((f'unit{k}.cu', private + secondary + f'\n// STEP17 relocatable unit {k}\n' + body))
    return sources, dict(stats, units=len(sources), unit_weights=loads, internal_linkage=internal,
                         private_helpers=[] if internal else renamed)


def _internalize():
    return os.environ.get('LATCH_STEP17_INTERNALIZE', 'on').lower() in {'1', 'on', 'true', 'yes'}


def _internalized_unit(secondary, texts, system_includes, k):
    """Secondary unit whose helpers have internal linkage (as in whole-program compilation).

    System headers are included first, outside; the program prefix and the chunk bodies live in an unnamed
    namespace so headers' templates/noinline helpers are internal to this unit. Namespace-scope extern
    __shared__ declarations stay outside and resolve to unit 0's definitions. Each chunk is exported by a
    thin external wrapper that calls its internal body.
    """
    outside, inside = [], []
    for line in secondary.split('\n'):
        (outside if line.startswith('extern __shared__ ') else inside).append(line)
    impls, wrappers = [], []
    for text in texts:
        name = re.search(r'void (latch_s17_block_\w+)\(', text).group(1)
        head = text.split('{', 1)[0].strip()
        params = head[head.index('(') + 1:head.rindex(')')]
        names = [p.strip().split()[-1].lstrip('*&') for p in params.split(',')]
        impls.append(text.replace(f'static __device__ __noinline__ void {name}(',
                                  f'static __device__ __forceinline__ void {name}_impl(', 1))
        wrappers.append(f'__device__ __noinline__ void {name}({params}) {{ {name}_impl({", ".join(names)}); }}\n')
    return (system_includes + '\n'.join(outside) + f'\n// STEP17 relocatable unit {k}: internal helpers\nnamespace {{\n'
            + '\n'.join(inside) + '\n' + ''.join(impls) + '}  // namespace\n' + ''.join(wrappers))


def _header_texts(prefix, header_names):
    from .lower import _header_path
    seen, texts = set(), [prefix]
    pending = [n for n in header_names if n.endswith('.cuh')] + re.findall(r'#include "([\w.]+\.cuh)"', prefix)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            text = _header_path(name).read_text()
        except Exception:
            continue
        texts.append(text)
        pending += re.findall(r'#include "([\w.]+\.cuh)"', text)
    return texts


_NOINLINE = re.compile(r'(?:__device__\s+__noinline__|__noinline__\s+__device__)\s+[\w:<>,\s\*&]*?\b(\w+)\s*\(')


def _external_noinline_names(prefix, header_names):
    """Non-template, non-static __noinline__ device functions in the prefix or (transitively) included headers.

    nvlink refuses duplicate external definitions across relocatable units; secondary units rename them so
    each unit links its own identical copy (these helpers carry no state). Unit 0 keeps original names.
    """
    from .lower import _header_path
    seen, texts = set(), [prefix]
    pending = [n for n in header_names if n.endswith('.cuh')] + re.findall(r'#include "([\w.]+\.cuh)"', prefix)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            text = _header_path(name).read_text()
        except Exception:
            continue
        texts.append(text)
        pending += re.findall(r'#include "([\w.]+\.cuh)"', text)
    names = set()
    for text in texts:
        for match in _NOINLINE.finditer(text):
            line_start = text.rfind('\n', 0, match.start()) + 1
            before = text[max(0, match.start() - 200):match.start()]
            if 'static' in text[line_start:match.start()] or re.search(r'template\s*<[^;{}]*>\s*$', before):
                continue
            names.add(match.group(1))
    return sorted(names)


def _build_partitioned(program, directory, arch, device_flags, nvcc, units, jobs, cubin, timings):
    """Compile relocatable units in parallel and nvlink them; on any failure return a failed record."""
    from concurrent.futures import ThreadPoolExecutor
    record = dict(version=PARTITION_VERSION, requested_units=units)
    try:
        sources, stats = partition_sources(program.source, program.manifest.get('header_hashes', {}), units)
    except Unsupported as why:
        record.update(failed=True, reason=str(why))
        return record
    record.update(stats)
    folder = directory / 'units'
    folder.mkdir(exist_ok=True)
    for name, text in sources:
        (folder / name).write_text(text)
    jobs = max(1, min(int(jobs or len(sources)), len(sources), os.cpu_count() or 1))
    unit_mode = os.environ.get('LATCH_STEP17_UNIT_FLAGS', '-rdc=true').split()
    commands = [[nvcc, *device_flags, f'-arch={arch}', *unit_mode, f'-I{directory}', '-cubin', name, '-o', name[:-3] + '.cubin']
                for name, _ in sources]

    def run(argv):
        begin = time.perf_counter()
        proc = subprocess.run(argv, cwd=folder, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return proc.returncode, proc.stdout, time.perf_counter() - begin

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(run, commands))
    timings['partition_compile'] = time.perf_counter() - started
    record.update(jobs=jobs, unit_seconds=[r[2] for r in results], unit_sum_seconds=sum(r[2] for r in results))
    log = ''.join(r[1] for r in results)
    if any(r[0] for r in results):
        record.update(failed=True, reason='unit compile failed', log=log[-8000:])
        return record
    link = [str(_cuda_home(nvcc) / 'bin' / 'nvlink'), f'-arch={arch}', '-o', str(cubin),
            *[name[:-3] + '.cubin' for name, _ in sources]]
    started = time.perf_counter()
    proc = subprocess.run(link, cwd=folder, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    timings['partition_link'] = time.perf_counter() - started
    log += proc.stdout
    if proc.returncode or not cubin.exists():
        record.update(failed=True, reason='nvlink failed', log=log[-8000:])
        if cubin.exists():
            cubin.unlink()
        return record
    record.update(commands=dict(units=commands, link=link), log=log)
    return record


def _run_measured(argv, cwd):
    """Run one compiler process; returns (returncode, output, wall s, child-tree peak RSS KiB, user s, sys s)."""
    begin = time.perf_counter()
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout.read()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    return (proc.returncode, output.decode(errors='replace'), time.perf_counter() - begin, usage.ru_maxrss,
            usage.ru_utime, usage.ru_stime)


class _MemorySampler:
    """Host-wide used memory (MemTotal - MemAvailable) peak while a build stage runs; Linux only."""

    def __init__(self, interval=0.5):
        import threading
        self.interval, self.peak_kib, self.samples = interval, 0, 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def used_kib():
        try:
            rows = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            return int(rows['MemTotal'].split()[0]) - int(rows['MemAvailable'].split()[0])
        except (OSError, KeyError, ValueError):
            return None

    def run(self):
        while not self.stop.is_set():
            used = self.used_kib()
            if used is not None:
                self.peak_kib, self.samples = max(self.peak_kib, used), self.samples + 1
            self.stop.wait(self.interval)

    def __enter__(self):
        self.baseline_kib = self.used_kib()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def _unit_key(text, include_hashes, argv_flags, nvcc_version):
    material = json.dumps(dict(text=hashlib.sha256(text.encode()).hexdigest(), headers=include_hashes,
                               flags=argv_flags, nvcc=nvcc_version), sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def _build_intervals(program, directory, arch, device_flags, nvcc, settings, jobs, cubin, timings):
    """Interval partition: units compiled concurrently (optionally reused from a unit cache), then nvlink."""
    from concurrent.futures import ThreadPoolExecutor
    from . import step17_intervals as intervals
    units = int(settings['units'])
    keep_inline = tuple(settings['keep_inline'])
    record = dict(settings, mode='intervals')
    record.pop('unit_cache', None)
    started = time.perf_counter()
    headers = program.manifest.get('header_hashes', {})
    try:
        if units == 1:
            text, stats = intervals.outlined_program_source(program.source, headers, keep_inline=keep_inline,
                                                            split_bytes=settings.get('split_bytes'),
                                                            isolate=tuple(settings.get('isolate', ())),
                                                            shared_pool=bool(settings.get('shared_pool')))
            sources = [('unit0.cu', text)]
        else:
            sources, stats = intervals.partition_sources(program.source, headers, units, keep_inline=keep_inline,
                                                         split_bytes=settings.get('split_bytes'),
                                                         isolate=tuple(settings.get('isolate', ())),
                                                         shared_pool=bool(settings.get('shared_pool')))
    except intervals.Refusal as why:
        raise RuntimeError('interval partition refused: ' + str(why)) from None
    folder = directory / 'units'
    folder.mkdir(exist_ok=True)
    for name, text in sources:
        (folder / name).write_text(text)
    prefix = program.source.split('extern "C" __global__')[0]
    include_hashes = {name: hashlib.sha256(text.encode()).hexdigest()
                      for name, text in intervals._header_names(prefix, headers)}
    try:
        patched = intervals.unit_headers(prefix, headers)
    except intervals.Refusal as why:
        raise RuntimeError('interval partition refused: ' + str(why)) from None
    for name, text in patched.items():
        (folder / name).write_text(text)   # quoted includes from the unit sources find this copy first
        include_hashes[name] = hashlib.sha256(text.encode()).hexdigest()
    record['patched_unit_headers'] = {name: include_hashes[name] for name in patched}
    timings['interval_source_preparation'] = time.perf_counter() - started
    record.update(stats, include_hashes=include_hashes)
    unit_flags = list(settings['unit_flags']) if units > 1 else []
    commands = [[nvcc, *device_flags, f'-arch={arch}', *unit_flags, f'-I{directory}', '-cubin', name,
                 '-o', name[:-3] + '.cubin'] for name, _ in sources]
    nvcc_version = subprocess.run([nvcc, '--version'], text=True, capture_output=True, check=True).stdout
    cache_root = settings.get('unit_cache')
    keys = [_unit_key(text, include_hashes, command[1:-5] + [command[-4]], nvcc_version)
            for (_, text), command in zip(sources, commands)]
    hits = []
    for (name, _), key in zip(sources, keys):
        cached = Path(cache_root) / 'units' / (key + '.cubin') if cache_root else None
        if cached is not None and cached.exists():
            shutil.copyfile(cached, folder / (name[:-3] + '.cubin'))
            hits.append(name)
    pending = [(name, command, key) for (name, _), command, key in zip(sources, commands, keys) if name not in hits]
    if units > 1:
        # Keep each unit's PTX (no extra compilation) to check the variables it defines (step17_intervals.unit_state).
        for k, (name, command, key) in enumerate(pending):
            keep = folder / ('keep_' + name[:-3])
            keep.mkdir(exist_ok=True)
            pending[k] = (name, command[:-4] + ['-keep', f'-keep-dir={keep.name}'] + command[-4:], key)
    jobs = max(1, min(int(jobs or len(pending) or 1), max(1, len(pending))))
    started = time.perf_counter()
    cuobjdump = str(_cuda_home(nvcc) / 'bin' / 'cuobjdump')

    def compile_and_guard(row):
        # LATCH-COLD-PREP: separately compiled attention items can expose ptxas's loop-carried wgmma register-A
        # clobber. Relocatable units are checked as soon as they compile (nvlink only resolves relocations).
        result = _run_measured(row[1], folder)
        if result[0] or units == 1:
            return result, None, 0.0
        from .wgmma_guard import hazards
        begin = time.perf_counter()
        sass = subprocess.run([cuobjdump, '-sass', row[0][:-3] + '.cubin'], cwd=folder, capture_output=True,
                              text=True, check=True).stdout
        return result, hazards(sass), time.perf_counter() - begin

    with _MemorySampler() as sampler, ThreadPoolExecutor(max_workers=jobs) as pool:
        guarded = list(pool.map(compile_and_guard, pending))
    results = [row[0] for row in guarded]
    unit_hazards = {name: row[1] for (name, _, _), row in zip(pending, guarded) if row[1]}
    timings['unit_compile_wall'] = time.perf_counter() - started
    rows = {}
    for (name, command, key), (rc, out, seconds, rss, user, system) in zip(pending, results):
        rows[name] = dict(seconds=seconds, max_rss_kib=rss, user_seconds=user, sys_seconds=system,
                          returncode=rc, key=key, bytes=len(dict(sources)[name]))
    for (name, _, _), row in zip(pending, guarded):
        rows[name]['wgmma_guard_seconds'] = row[2]
        (folder / (name[:-3] + '.log')).write_text(row[0][1])
    log = ''.join(f'==== {name}\n' + (folder / (name[:-3] + '.log')).read_text() for name, _, _ in pending)
    record['ptxas_wgmma_serialized'] = sorted(set(re.findall(
        r"\((C75\d\d)\) Potential Performance Loss: wgmma\.mma_async instructions are serialized due to ([^']*?) "
        r"in the function '([^']+)'", log)))
    record.update(jobs=jobs, unit_rows=rows, unit_cache_hits=hits, unit_cache_keys=dict(zip(dict(sources), keys)),
                  compile_host_used_peak_kib=sampler.peak_kib, compile_host_used_baseline_kib=sampler.baseline_kib,
                  unit_sum_seconds=sum(r['seconds'] for r in rows.values()),
                  unit_cpu_seconds=sum(r['user_seconds'] + r['sys_seconds'] for r in rows.values()))
    failed = [name for name, row in rows.items() if row['returncode']]
    if failed:
        (directory / 'build.log').write_text(log)
        (directory / 'intervals.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        raise RuntimeError(f'interval unit compile failed: {failed}; preserved {folder}\n{log[-8000:]}')
    if unit_hazards:
        record['wgmma_guard_hazards'] = {name: rows_[:5] for name, rows_ in unit_hazards.items()}
        (directory / 'intervals.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        raise RuntimeError(f'units with loop-carried wgmma register-A hazards: {sorted(unit_hazards)}')
    if units > 1:
        state = {}
        for name, _, _ in pending:
            keep = folder / ('keep_' + name[:-3])
            if name != 'unit0.cu':   # unit0 owns every external definition; cached units passed when compiled
                ptx = sorted(keep.glob('*.ptx'))
                if not ptx:
                    raise RuntimeError(f'no kept PTX for {name}')
                state[name] = intervals.unit_state(''.join(path.read_text() for path in ptx))
            shutil.rmtree(keep)
        record['secondary_unit_state'] = state
        refused = {name: row['refused'] for name, row in state.items() if row['refused']}
        if refused:
            (directory / 'intervals.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
            raise RuntimeError(f'secondary units define private global/constant variables: {refused}')
    if cache_root:
        (Path(cache_root) / 'units').mkdir(parents=True, exist_ok=True)
        for name, _, key in pending:
            target = Path(cache_root) / 'units' / (key + '.cubin')
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            shutil.copyfile(folder / (name[:-3] + '.cubin'), temporary)
            temporary.replace(target)
    if units == 1:
        shutil.copyfile(folder / 'unit0.cubin', cubin)
        link = None
    else:
        link = [str(_cuda_home(nvcc) / 'bin' / 'nvlink'), f'-arch={arch}', '-o', str(cubin),
                *[name[:-3] + '.cubin' for name, _ in sources]]
        started = time.perf_counter()
        rc, out, seconds, rss, user, system = _run_measured(link, folder)
        timings['nvlink'] = seconds
        record.update(nvlink=dict(seconds=seconds, max_rss_kib=rss, user_seconds=user, sys_seconds=system,
                                  returncode=rc))
        log += '==== nvlink\n' + out
        if rc or not cubin.exists():
            (directory / 'build.log').write_text(log)
            raise RuntimeError(f'nvlink failed ({rc}); preserved {folder}\n{out[-8000:]}')
    found = []
    if units == 1:
        # One whole-program cubin: guard it once (relocatable units were guarded as they compiled).
        from .wgmma_guard import hazards
        started = time.perf_counter()
        found = hazards(subprocess.run([cuobjdump, '-sass', str(cubin)], capture_output=True, text=True,
                                       check=True).stdout)
        timings['wgmma_guard'] = time.perf_counter() - started
    record['wgmma_guard_hazards'] = found[:20]
    record['wgmma_guard_hazard_count'] = len(found) + sum(len(v) for v in unit_hazards.values())
    (directory / 'intervals.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    if found:
        cubin.unlink()
        raise RuntimeError(f'linked SASS has {len(found)} loop-carried wgmma register-A hazards: {found[:3]}')
    record.update(commands=dict(units=[row[1] for row in pending] if pending else commands, link=link), log=log)
    return record
