"""Release-identity programs: release operations lowered with release options only (CPU; no CUDA, no nvcc).

    PYTHONPATH=<release checkout>/src python tests/operations/release_identity_cases.py \
        --write tests/operations/fixtures/release_program_hashes.json
    python tests/operations/release_identity_cases.py --check tests/operations/fixtures/release_program_hashes.json

Every case captures a small program built only from operations the release already had (segmented products,
row permutations, cat/slice copies, embedding gathers, row reductions, softmaxes, the vocabulary head, split-K linear
products) and lowers it with
``lower(..., target_arch="sm_90")`` inside the option scopes ``prepare_module`` opens, setting only options the release
already accepted. Option families added since the release (``segmented_attention``, ``tile_readiness``,
``vocab_rules``, ``elementwise_rules``, ``graph_rules`` and the newer tokens of the existing families) stay unset, so
the generated ``program.cu`` must equal the release's byte for byte. The hashes in the fixture are produced by running
this script against the release tree (``--write``); ``test_release_identity.py`` lowers the same cases with the current
tree and compares. The cases cover the release behaviour that the opt-in rules change when selected:

* ``claimed_experts`` -- claimed SwiGLU expert products (N-major stacked banks, paired gate/up products, staged
  epilogues) with the release accumulation, matrix-body and update-schedule rules;
* ``claimed_tiles`` -- an N-major product with K % 64 != 0, a K-major claimed product with N = 160 and claimed weight
  gradients with fewer than 128 tasks at 128 x 128;
* ``split_counters`` -- long-K linear weight gradients on the split-K bodies in the flat dependency schedule;
* ``copies`` -- a row permutation of 16-byte-aligned rows and 2**20-element BF16 pad (cat) and FP32 slice_backward
  copies whose 28-byte rows the 16-byte copy form declines;
* ``shared_casts`` -- one FP32 embedding gather read by two BF16 casts that feed segmented products;
* ``row_chain`` -- an embedding gather, two chained RMSNorm-style row stages, a log_softmax cross-entropy and an
  argmax over a 48-entry vocabulary, with the release ``update`` accumulation preset;
* ``row_softmax`` -- row softmaxes at width 48 and width 5 with their backward, 2048 rows, ``update`` preset;
* ``vocab_head`` -- the release vocabulary-head rewrite (``vocabulary_head="force"``) of a BF16 linear head read by
  ``F.cross_entropy``, with its backward;
* ``mutable_roots`` -- explicit model state, a cat of two BF16 projections and a full mean whose scratch value is
  written twice, so the effect order on a root with repeated writes is part of the program;
* ``p6d_training`` -- the BF16-projection training graph of ``test_p6d_retention``, whose normal compose selection
  retains the P6D schedule (its exact reductions come from ``codegen_rules``), under both precision policies.

Captured graphs depend on the torch release, so the hashes hold for the torch base version recorded in the fixture
(2.9.1). The local build tag (+cu128, +cpu) is ignored: capture and lowering run on the CPU and are not expected
to depend on it. Environment
variables with the prefixes in ``ENVIRONMENT_PREFIXES`` are cleared before lowering.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch

RELEASE_COMMIT = "9bb7b3343"
SCHEMA = "latch-release-program-hashes-v1"
FIXTURE = Path(__file__).with_name("fixtures") / "release_program_hashes.json"
# Option families the release accepted, in prepare_module's scope order.
RELEASE_OPTIONS = ("update_schedule", "segmented_products", "norm_fusion", "attention_rules", "matrix_schedule",
                   "matrix_bodies", "accumulation_rules", "memory_rules", "exact_bf16_products")
# Option families added after the release; entered unset when the tree has them (prepare_module's default).
NEWER_OPTIONS = ("segmented_attention_rules", "tile_readiness", "vocab_rules", "elementwise_rules", "graph_rules")
CLAIMED_PRODUCTS = "flat+tma+rotate+claim+sepi"
# Prefixes of the environment variables the compiler package reads; several of them change lowering, so the cases are
# lowered with none of them set.
ENVIRONMENT_PREFIXES = ("LATCH_", "P8C_", "P9C_")
# Set to 1 to fail, instead of skip, when the installed torch differs from the one the hashes were recorded with.
REQUIRE_VARIABLE = "LATCH_REQUIRE_RELEASE_IDENTITY"


def lowering_environment(environ=None):
    """Names of the set environment variables that can change the generated program."""
    environ = os.environ if environ is None else environ
    return sorted(name for name in environ if name.startswith(ENVIRONMENT_PREFIXES) and name != REQUIRE_VARIABLE)


def torch_base(version):
    """The torch release a version string names, without its local build tag ('2.9.1+cu128' -> '2.9.1')."""
    return str(version or "").split("+")[0]


def _parameter(generator, *shape, scale=1.0):
    return torch.nn.Parameter(torch.randn(*shape, generator=generator) * scale)


def _optimizer():
    from megakernel_lab.operation_compiler.optim import SGDConfig
    return SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0)


# ----------------------------------------------------------------------------------------------- programs
class ClaimedExperts(torch.nn.Module):
    """SwiGLU experts over N-major stacked banks [G, K, N]; gate and up read the same rows (paired products)."""

    def __init__(self, groups=4, rows=4352, width=256, hidden=384):
        super().__init__()
        generator = torch.Generator().manual_seed(9101)
        self.x = _parameter(generator, rows, width, scale=0.5)
        self.gate = _parameter(generator, groups, width, hidden, scale=0.05)
        self.up = _parameter(generator, groups, width, hidden, scale=0.05)
        self.down = _parameter(generator, groups, hidden, width, scale=0.05)

    def forward(self, offsets):
        from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
        x = self.x.to(torch.bfloat16)
        gate = ragged_mm(x, self.gate.to(torch.bfloat16), offsets)
        up = ragged_mm(x, self.up.to(torch.bfloat16), offsets)
        hidden = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
        return ragged_mm(hidden, self.down.to(torch.bfloat16), offsets)


class ClaimedTiles(torch.nn.Module):
    """An N-major bank with K % 64 == 32 and a K-major bank with N = 160 (few-task weight gradients)."""

    def __init__(self, groups=4, rows=4352, tail=352, width=256, columns=160):
        super().__init__()
        generator = torch.Generator().manual_seed(9102)
        self.a = _parameter(generator, rows, tail, scale=0.5)
        self.tail_bank = _parameter(generator, groups, tail, width, scale=0.05)       # [G, K, N], N-major
        self.b = _parameter(generator, rows, width, scale=0.5)
        self.column_bank = _parameter(generator, groups, columns, width, scale=0.05)  # read as [G, K, N], K-major

    def forward(self, offsets):
        from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
        first = ragged_mm(self.a.to(torch.bfloat16), self.tail_bank.to(torch.bfloat16), offsets)
        second = ragged_mm(self.b.to(torch.bfloat16), self.column_bank.transpose(1, 2).to(torch.bfloat16), offsets)
        return first, second


class SplitLinear(torch.nn.Module):
    """A 3-wide input and a 1-wide head: both weight gradients contract over the 2048-row batch (split-K)."""

    def __init__(self, inputs=3, width=64, outputs=1):
        super().__init__()
        generator = torch.Generator().manual_seed(9103)
        self.first = _parameter(generator, inputs, width, scale=0.3)
        self.head = _parameter(generator, width, outputs, scale=0.1)

    def forward(self, x):
        hidden = torch.nn.functional.silu((x.to(torch.bfloat16) @ self.first.to(torch.bfloat16)).float())
        return (hidden.to(torch.bfloat16) @ self.head.to(torch.bfloat16)).float()


class Copies(torch.nn.Module):
    """A row permutation of 512-byte BF16 rows; 2**20-element BF16 pad and FP32 slice_backward copies (28-byte rows)."""

    def __init__(self, rows=1031, width=256, items=16384, heads=4, inner=14):
        super().__init__()
        generator = torch.Generator().manual_seed(9104)
        self.x, self.wx = _parameter(generator, rows, width), _parameter(generator, rows, width)
        self.a, self.wa = _parameter(generator, items, heads, inner), _parameter(generator, items, heads, inner + 2)
        self.t, self.wt = _parameter(generator, items, heads, inner + 2), _parameter(generator, items, heads, inner)
        self.inner = inner

    def forward(self, order, inverse):
        from megakernel_lab.operation_compiler.row_permutation import permutation_gather
        rows = permutation_gather(self.x.to(torch.bfloat16), order, inverse)
        a = self.a.to(torch.bfloat16)
        pad = torch.cat((a, a.new_zeros(*a.shape[:-1], 2)), -1)
        sliced = self.t[..., :self.inner]
        return ((rows.float() * self.wx).sum() + (pad.float() * self.wa).sum() + (sliced * self.wt).sum())


class SharedCasts(torch.nn.Module):
    """One FP32 embedding gather read by two identical BF16 casts, each the A operand of a segmented product."""

    def __init__(self, vocab=512, width=128, outputs=128, groups=4):
        super().__init__()
        generator = torch.Generator().manual_seed(9105)
        self.table = _parameter(generator, vocab, width, scale=0.5)
        self.first = _parameter(generator, groups, outputs, width, scale=0.1)    # read as [G, K, N], K-major
        self.second = _parameter(generator, groups, outputs, width, scale=0.1)

    def forward(self, tokens, offsets):
        from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
        rows = torch.nn.functional.embedding(tokens, self.table)
        first = ragged_mm(rows.to(torch.bfloat16), self.first.transpose(1, 2).to(torch.bfloat16), offsets)
        second = ragged_mm(rows.to(torch.bfloat16), self.second.transpose(1, 2).to(torch.bfloat16), offsets)
        return (first.float() * second.float()).mean()


class SharedProjections(torch.nn.Module):
    """The training graph of test_p6d_retention: one projection weight shared by three tanh layers."""

    def __init__(self):
        super().__init__()
        self.input = torch.nn.Parameter(torch.full((64, 1024), 0.01))
        self.weight = torch.nn.Parameter(torch.full((1024, 1024), 0.001))
        self.bias = torch.nn.Parameter(torch.zeros(1024))
        self.head = torch.nn.Parameter(torch.full((1024, 64), 0.01))

    def forward(self, x):
        h = torch.tanh(x @ self.input)
        for _ in range(3):
            h = torch.tanh(h @ self.weight + self.bias) + 0.2 * h
        return h @ self.head


class RowChain(torch.nn.Module):
    """Embedding gather, two chained RMSNorm-style row stages (mean(x*x)), then a small-vocabulary head read by a
    log_softmax cross-entropy and an argmax."""

    def __init__(self, vocab=48, width=96):
        super().__init__()
        generator = torch.Generator().manual_seed(9106)
        self.table = _parameter(generator, vocab, width, scale=0.5)
        self.scale = torch.nn.Parameter(torch.ones(width))
        self.head = _parameter(generator, width, vocab, scale=0.1)

    def forward(self, tokens):
        x = torch.nn.functional.embedding(tokens, self.table)
        h = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * self.scale
        h = h + x
        h = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6)
        return h @ self.head


def _row_chain_loss(logits, targets):
    picked = torch.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1))
    hits = (logits.argmax(-1) == targets).float().mean()
    return -picked.mean() + 0.0 * hits


class RowSoftmax(torch.nn.Module):
    """Row softmaxes at width 48 (32 < W <= 256, W % 32 != 0) and width 5 (W <= 16), each with its backward."""

    def __init__(self, rows=2048):
        super().__init__()
        generator = torch.Generator().manual_seed(9107)
        self.mid, self.mid_weight = _parameter(generator, rows, 48), _parameter(generator, rows, 48)
        self.narrow, self.narrow_weight = _parameter(generator, rows, 5), _parameter(generator, rows, 5)

    def forward(self):
        mid = torch.softmax(self.mid, -1) * self.mid_weight
        narrow = torch.softmax(self.narrow, -1) * self.narrow_weight
        return mid.sum() + narrow.sum()


class VocabHead(torch.nn.Module):
    """A BF16 linear vocabulary head (vocabulary 1013, padded by the release head rewrite) for F.cross_entropy."""

    def __init__(self, width=64, vocab=1013):
        super().__init__()
        generator = torch.Generator().manual_seed(9108)
        self.weight = _parameter(generator, width, vocab, scale=width ** -0.5)

    def forward(self, hidden):
        return (hidden.to(torch.bfloat16) @ self.weight.to(torch.bfloat16)).float()


class MutableRoots(torch.nn.Module):
    """Explicit model state updated each step and two projections written as pieces of one cat buffer."""

    def __init__(self, rows=256, width=64):
        super().__init__()
        generator = torch.Generator().manual_seed(9109)
        self.first = _parameter(generator, width, width, scale=0.1)
        self.second = _parameter(generator, width, width, scale=0.1)
        self.mix = _parameter(generator, 2 * width, width, scale=0.1)

    def forward(self, x, state):
        x = x.to(torch.bfloat16)
        both = torch.cat((x @ self.first.to(torch.bfloat16), x @ self.second.to(torch.bfloat16)), -1)
        y = (torch.tanh(both.float()).to(torch.bfloat16) @ self.mix.to(torch.bfloat16)).float()
        mean = 0.9 * state["mean"] + 0.1 * y.detach().mean(0)
        return y - mean, {"mean": mean}


def _offsets(rows=4352):
    return torch.tensor([0, 1000, 2200, 3300, rows], dtype=torch.int64)


def _permutation(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(rows, generator=generator)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(rows)
    return order, inverse


def _claimed_experts():
    return dict(module=ClaimedExperts(), inputs=(_offsets(),), loss=lambda y: y.float().square().mean(),
                options=dict(segmented_products=CLAIMED_PRODUCTS, accumulation_rules="update+fsload+fsspread",
                             matrix_bodies="exact", update_schedule="streamed_prep"),
                precision_policy="practical")


def _claimed_tiles():
    return dict(module=ClaimedTiles(), inputs=(_offsets(),),
                loss=lambda y: y[0].float().square().mean() + y[1].float().square().mean(),
                options=dict(segmented_products=CLAIMED_PRODUCTS), precision_policy="practical")


def _split_counters():
    generator = torch.Generator().manual_seed(9113)
    x, y = torch.randn(2048, 3, generator=generator), torch.randn(2048, 1, generator=generator)
    return dict(module=SplitLinear(), inputs=(x,), targets=(y,), loss=lambda p, t: (p - t).square().mean())


def _copies():
    return dict(module=Copies(), inputs=_permutation(1031, 9114), loss=lambda y: y)


def _shared_casts():
    tokens = torch.randint(0, 512, (256,), generator=torch.Generator().manual_seed(9115))
    offsets = torch.tensor([0, 50, 120, 200, 256], dtype=torch.int64)
    return dict(module=SharedCasts(), inputs=(tokens, offsets), loss=lambda y: y, precision_policy="practical")


def _row_chain():
    generator = torch.Generator().manual_seed(9116)
    tokens = torch.randint(0, 48, (512,), generator=generator)
    targets = torch.randint(0, 48, (512,), generator=generator)
    return dict(module=RowChain(), inputs=(tokens,), targets=(targets,), loss=_row_chain_loss,
                options=dict(accumulation_rules="update"))


def _row_softmax():
    return dict(module=RowSoftmax(), inputs=(), loss=lambda y: y, options=dict(accumulation_rules="update"))


def _vocab_head():
    generator = torch.Generator().manual_seed(9117)
    hidden = torch.randn(64, 64, generator=generator)
    targets = torch.randint(0, 1013, (64,), generator=generator)
    return dict(module=VocabHead(), inputs=(hidden,), targets=(targets,),
                loss=lambda logits, t: torch.nn.functional.cross_entropy(logits, t),
                lower_options=dict(vocabulary_head="force"))


def _mutable_roots():
    x = torch.randn(256, 64, generator=torch.Generator().manual_seed(9118))
    return dict(module=MutableRoots(), inputs=(x,), loss=lambda y: y.square().mean(),
                capture=dict(state={"mean": torch.zeros(64)}), precision_policy="practical")


def _p6d_training(policy):
    def case():
        from megakernel_lab.operation_compiler.precision import BF16_PROJECTIONS
        return dict(module=SharedProjections(), inputs=(torch.full((1280, 64), 0.1),),
                    targets=(torch.zeros((1280, 64)),), loss=lambda y, t: (y - t).square().sum(-1).mean(),
                    capture=dict(precision=BF16_PROJECTIONS), optimizer=False,
                    lower=dict(execution="compose", precision_policy=policy))
    return case


CASES = {
    "claimed_experts": _claimed_experts,
    "claimed_tiles": _claimed_tiles,
    "split_counters": _split_counters,
    "copies": _copies,
    "shared_casts": _shared_casts,
    "row_chain": _row_chain,
    "row_softmax": _row_softmax,
    "vocab_head": _vocab_head,
    "mutable_roots": _mutable_roots,
    "p6d_training_strict": _p6d_training("strict"),
    "p6d_training_practical": _p6d_training("practical"),
}


# ----------------------------------------------------------------------------------------------- lowering
@contextmanager
def option_scopes(options):
    """The option scopes ``prepare_module`` opens around ``lower()``, with only release option families set."""
    unknown = set(options) - set(RELEASE_OPTIONS)
    if unknown:
        raise ValueError(f"release-identity cases may set release options only, not {sorted(unknown)}")
    from megakernel_lab.operation_compiler import (accumulation_rules, attention_rules, matrix_bodies,
                                                   matrix_schedule, memory_rules, norm_fusion, ragged_experts_lower,
                                                   update_schedule)
    from megakernel_lab.operation_compiler.lower import exact_bf16_products_scope
    with ExitStack() as stack:
        stack.enter_context(update_schedule.scope(options.get("update_schedule")))
        stack.enter_context(ragged_experts_lower.products_scope(options.get("segmented_products")))
        stack.enter_context(norm_fusion.scope(options.get("norm_fusion")))
        stack.enter_context(attention_rules.scope(options.get("attention_rules")))
        stack.enter_context(matrix_schedule.scope(options.get("matrix_schedule")))
        stack.enter_context(matrix_bodies.scope(options.get("matrix_bodies")))
        stack.enter_context(accumulation_rules.scope(options.get("accumulation_rules")))
        stack.enter_context(memory_rules.scope(options.get("memory_rules")))
        stack.enter_context(exact_bf16_products_scope(options.get("exact_bf16_products")))
        for name in NEWER_OPTIONS:
            qualified = f"megakernel_lab.operation_compiler.{name}"
            try:
                module = importlib.import_module(qualified)
            except ModuleNotFoundError as error:
                if error.name != qualified:
                    raise
                continue                                    # the release tree: the family does not exist yet
            stack.enter_context(module.scope(None))
        yield


def lower_case(name):
    """The GeneratedProgram of one case, lowered as prepare_module lowers (CPU only)."""
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    case = CASES[name]()
    extra = dict(case.get("capture", {}))
    if case.get("optimizer", True):
        extra["optimizer"] = _optimizer()
    captured = capture_module(case["module"], case["loss"], case["inputs"], case.get("targets", ()), **extra)
    with option_scopes(case.get("options", {})):
        if "lower" in case:                                 # the plain lower() call of the case's own test
            return lower(captured.ir, **case["lower"])
        return lower(captured.ir, retain_all=False, policy="auto", target_arch="sm_90",
                     precision_policy=case.get("precision_policy"), compact_codegen="off", execution="compose",
                     compose_selection="auto", recurrent_gradients="inline", **case.get("lower_options", {}))


def program_hash(name):
    """sha256 of the generated program.cu of one case."""
    return hashlib.sha256(lower_case(name).source.encode()).hexdigest()


def load_hashes(path=FIXTURE):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path, help="record the hashes of the importable tree (run against the release)")
    mode.add_argument("--check", type=Path, help="compare the importable tree against recorded hashes")
    parser.add_argument("--case", action="append", choices=sorted(CASES), help="only these cases (repeatable)")
    args = parser.parse_args(argv)
    cleared = lowering_environment()
    for name in cleared:
        os.environ.pop(name)
    if cleared:
        print(f"lowering with these environment variables unset: {', '.join(cleared)}", file=sys.stderr)
    torch.set_num_threads(1)
    names = args.case or list(CASES)
    hashes = {}
    for name in names:
        hashes[name] = program_hash(name)
        print(f"{name} {hashes[name]}", flush=True)
    if args.write:
        record = load_hashes(args.write) if args.write.exists() and args.case else {}
        record.update(schema=SCHEMA, release=RELEASE_COMMIT, generator="tests/operations/release_identity_cases.py",
                      torch=torch.__version__)
        record["cases"] = {**record.get("cases", {}), **hashes}
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return 0
    recorded = load_hashes(args.check).get("cases", {})
    differ = sorted(name for name in names if recorded.get(name) != hashes[name])
    if differ:
        print(f"program.cu differs from the release for: {', '.join(differ)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
