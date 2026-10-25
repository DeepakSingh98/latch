"""Native per-launch binding check for Runtime (LATCH-SELECTIVE-10).

``Runtime`` refuses a launch when any bound tensor's device address differs from its admitted address (a replaced or
resized storage would make the kernel's pointer table stale). The Python comparison costs ~0.3 ms of idle-GPU host
time per launch for ~3,300 bound tensors. This module compiles (once per torch build, with the host C++ compiler
and torch's own headers; no ninja) a TorchScript custom class that holds the bound tensors -- the same TensorImpls
as the Python objects, so ``.data =``, ``set_`` and ``resize_`` are visible to it -- and compares their current
``data_ptr()`` with the admitted addresses in one C++ loop. Same refusal; any failure to build or load leaves the
caller on its Python fallback.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = r'''
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

struct LatchBindingGuard : torch::CustomClassHolder {
  std::vector<at::Tensor> tensors;
  std::vector<int64_t> pointers;
  explicit LatchBindingGuard(std::vector<at::Tensor> bound) : tensors(std::move(bound)) {
    pointers.reserve(tensors.size());
    for (const auto& t : tensors) pointers.push_back(reinterpret_cast<int64_t>(t.data_ptr()));
  }
  bool check() const {
    for (size_t i = 0; i < tensors.size(); ++i)
      if (reinterpret_cast<int64_t>(tensors[i].data_ptr()) != pointers[i]) return false;
    return true;
  }
  int64_t size() const { return static_cast<int64_t>(tensors.size()); }
};

TORCH_LIBRARY(latch_binding_guard_v1, m) {
  m.class_<LatchBindingGuard>("BindingGuard")
      .def(torch::init<std::vector<at::Tensor>>())
      .def("check", &LatchBindingGuard::check)
      .def("size", &LatchBindingGuard::size);
}
'''

VERSION_SOURCE = r'''
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

struct LatchVersionGuard : torch::CustomClassHolder {
  std::vector<at::Tensor> tensors;
  std::vector<int64_t> versions;
  explicit LatchVersionGuard(std::vector<at::Tensor> sources) : tensors(std::move(sources)) { sync(); }
  void sync() {
    versions.clear();
    for (const auto& t : tensors) versions.push_back(static_cast<int64_t>(t._version()));
  }
  bool changed() const {
    for (size_t i = 0; i < tensors.size(); ++i)
      if (static_cast<int64_t>(tensors[i]._version()) != versions[i]) return true;
    return false;
  }
};

TORCH_LIBRARY(latch_version_guard_v1, m) {
  m.class_<LatchVersionGuard>("VersionGuard")
      .def(torch::init<std::vector<at::Tensor>>())
      .def("sync", &LatchVersionGuard::sync)
      .def("changed", &LatchVersionGuard::changed);
}
'''

_LOADED = None


def _cache_dir():
    root = os.environ.get("LATCH_BINDING_GUARD_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "latch_binding_guard"


def _load(source, stem, library, cls):
    import torch
    from torch.utils import cpp_extension
    compiler = shutil.which(os.environ.get("CXX", "g++")) or shutil.which("c++")
    if compiler is None:
        return None
    key = hashlib.sha256((source + torch.__version__ + str(torch._C._GLIBCXX_USE_CXX11_ABI)).encode()).hexdigest()[:24]
    target = _cache_dir() / f"{stem}_{key}.so"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            src = Path(temp) / "guard.cpp"
            src.write_text(source)
            out = Path(temp) / "guard.so"
            includes = [f"-I{p}" for p in cpp_extension.include_paths()]
            libdir = Path(torch.__file__).parent / "lib"
            command = [compiler, "-O2", "-shared", "-fPIC", "-std=c++17", "-w",
                       f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", *includes, str(src),
                       f"-L{libdir}", "-lc10", "-ltorch_cpu", f"-Wl,-rpath,{libdir}", "-o", str(out)]
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            os.replace(out, target)
    torch.ops.load_library(str(target))
    return getattr(getattr(torch.classes, library), cls)


def guard_class():
    """torch.classes.latch_binding_guard_v1.BindingGuard, or None when it cannot be built or loaded."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED or None
    _LOADED = False
    try:
        _LOADED = _load(SOURCE, "latch_binding_guard", "latch_binding_guard_v1", "BindingGuard") or False
    except Exception:  # fall back to the Python comparison
        _LOADED = False
    return _LOADED or None


_VERSION_LOADED = None


def version_guard_class():
    """torch.classes.latch_version_guard_v1.VersionGuard (dcast source-parameter versions), or None."""
    global _VERSION_LOADED
    if _VERSION_LOADED is not None:
        return _VERSION_LOADED or None
    _VERSION_LOADED = False
    try:
        _VERSION_LOADED = _load(VERSION_SOURCE, "latch_version_guard", "latch_version_guard_v1", "VersionGuard") or False
    except Exception:  # fall back to the Python comparison
        _VERSION_LOADED = False
    return _VERSION_LOADED or None
