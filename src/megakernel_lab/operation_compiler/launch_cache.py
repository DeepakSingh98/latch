"""Per-bound-runtime launch admission, with no process-global mutable cache.

Integration adaptation of the host-overhead finding in C-OPS-I
f121882295ecdcf094f8727e8d72f15fc51199ba. The donor's static device cache is
intentionally replaced by a context- and binding-specific runtime admission.

CUDA 12.8.1 documents cuCtxGetId(NULL, &id) as the current context's identity,
unique for the lifetime of the program, even when contexts are destroyed:
https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__CTX.html
The driver library is resolved per Runtime; missing support uses latch_launch.
"""
from __future__ import annotations

import ctypes


def admitted_launch_source(shared_bytes, cluster=0):
    """Stateless CUDA host export; caller owns and validates its admission."""
    if type(shared_bytes) is not int or shared_bytes <= 0:
        raise ValueError('shared_bytes must be a positive integer')
    if cluster:
        from .lower import _cluster_launch
        return ('''
extern "C" int latch_launch_admitted(float** args, float* arena, unsigned long long* ticks,
                                     int blocks, void* stream, int admitted_max_blocks,
                                     int admitted_device) {
  int device; cudaError_t e = cudaGetDevice(&device); if (e != cudaSuccess) return e;
  if (device != admitted_device) return cudaErrorInvalidDevice;
  if (blocks <= 0 || blocks > admitted_max_blocks) return cudaErrorInvalidConfiguration;
  void* kernel_args[] = {&args, &arena, &ticks};
''' + _cluster_launch(shared_bytes, cluster) + '''}
''')
    return '''
extern "C" int latch_launch_admitted(float** args, float* arena, unsigned long long* ticks,
                                     int blocks, void* stream, int admitted_max_blocks,
                                     int admitted_device) {
  int device; cudaError_t e = cudaGetDevice(&device); if (e != cudaSuccess) return e;
  if (device != admitted_device) return cudaErrorInvalidDevice;
  if (blocks <= 0 || blocks > admitted_max_blocks) return cudaErrorInvalidConfiguration;
  void* kernel_args[] = {&args, &arena, &ticks};
  return cudaLaunchCooperativeKernel((const void*)latch_ops, dim3(blocks), dim3(256),
                                    kernel_args, ''' + str(shared_bytes) + ''', (cudaStream_t)stream);
}
'''


class ContextIdentity:
    """A driver symbol bound to this instance, never a shared admission cache."""

    def __init__(self):
        self.library = ctypes.CDLL('libcuda.so.1')
        self.function = self.library.cuCtxGetId
        self.function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        self.function.restype = ctypes.c_int

    def __call__(self):
        value = ctypes.c_ulonglong()
        error = self.function(None, ctypes.byref(value))
        if error:
            raise RuntimeError(f'cuCtxGetId failed with CUDA driver error {error}')
        return value.value


def binding_identity(tensor):
    """Metadata only: value copies remain valid, storage/layout replacement does not."""
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, tensor.storage_offset(), tensor.is_neg(), tensor.is_conj())


def function_identity(function):
    return ctypes.cast(function, ctypes.c_void_p).value
