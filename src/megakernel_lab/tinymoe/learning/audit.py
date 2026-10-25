"""Independent CPU audits of complete updates from random weights (unchanged donor policy).

Policy: packaged megakernel_lab/tinymoe/POLICY.json `same_state_fixed_membership_complete_update`,
unchanged. Reference: installed Transformers 4.55.4 in FP64 with eager attention and FP64
RMS norms, the donor's independent expert loop at the observed memberships
(megakernel_lab.tinymoe.audit.force_membership), HF load-balancing aux over the pass.

  native      Latch B1/B2 observation: objective/gradients vs FP64 (CE_mean + 0.02 aux),
              routing interval check, activity check, FP64 clip + AdamW formula on the
              observed raw gradients with the run's staged hyper-parameters.
  microbatch  Arm A microbatch k: FP64 reference of (CE_mean + 0.02 aux)/16; writes ref_grads.pt.
  accumulate  Arm A: observed accumulated G vs the FP64 sum of the 16 reference gradients;
              FP64 clip + AdamW formula on G; experts unused in all 16 microbatches unchanged.
Gate scale: relative-L2 gates are scale free; arm-A microbatch objective/floor gates are
applied to 16x-normalized quantities, so they are no looser than unit-CE audits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from megakernel_lab.tinymoe.audit import POLICY_PATH, distance, force_membership


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def policy():
    raw = POLICY_PATH.read_bytes()
    return json.loads(raw)["same_state_fixed_membership_complete_update"], hashlib.sha256(raw).hexdigest()


def bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def reference(model_dir, parameters, tokens, targets, routes, ce_coefficient, router_coefficient):
    from megakernel_lab.tinymoe.model import load_model
    model = load_model(model_dir, attention="eager").double()
    if set(dict(model.named_parameters())) != set(parameters):
        raise ValueError("observation parameter inventory differs from installed model")
    model.load_state_dict(parameters, strict=False)
    force_membership(model, routes)
    output = model(input_ids=tokens, use_cache=False, output_router_logits=True)
    ce = F.cross_entropy(output.logits.reshape(-1, model.config.vocab_size), targets.reshape(-1))
    probabilities = [F.softmax(logits, dim=-1) for logits in output.router_logits]
    member = F.one_hot(routes.reshape(-1, routes.shape[-1]), model.num_experts).double().mean(0)
    auxiliary = model.num_experts * (member * torch.cat(probabilities, 0).mean(0)[None, :]).sum()
    loss = ce * ce_coefficient + router_coefficient * auxiliary
    loss.backward()
    grads = {n: (p.grad.detach() if p.grad is not None else torch.zeros_like(p)) for n, p in model.named_parameters()}
    return float(loss.detach()), grads, [p.detach() for p in probabilities]


def routing_rows(observed_probabilities, routes, reference_probabilities):
    rows = []
    for layer, expected in enumerate(reference_probabilities):
        actual = observed_probabilities[layer].double()
        error = (actual - expected).abs().amax(-1)
        k = routes.shape[-1]
        top = expected.topk(k + 1, dim=-1)
        disagreement = (routes[layer].sort(-1).values != top.indices[:, :k].sort(-1).values).any(-1)
        margin = top.values[:, k - 1] - top.values[:, k]
        outside = disagreement & (margin > 2 * error)
        rows.append(dict(layer=layer, tokens=int(routes.shape[1]), set_disagreements=int(disagreement.sum()),
                         outside_error_interval=int(outside.sum()), maximum_probability_error=float(error.max()),
                         minimum_cutoff_margin=float(margin.min())))
    return rows


def activity_rows(routes, active):
    rows = []
    for layer, selected in enumerate(routes):
        flags = active[f"active.{layer}"]
        counts = F.one_hot(selected.long(), flags.numel()).sum((0, 1))
        duplicate = bool((selected.sort(-1).values[:, 1:] == selected.sort(-1).values[:, :-1]).any())
        rows.append(dict(layer=layer, unused_experts=int((counts == 0).sum()),
                         passed=bool(torch.equal(flags > 0, counts > 0)) and not duplicate))
    return rows


def gradient_rows(observed, expected, gates, scale=1.0):
    rows = []
    for name, ref in expected.items():
        row = distance(observed[name].double() * scale, ref * scale, floor=gates["gradient_absolute_rms_floor"])
        row.update(name=name, passed=row["finite"] and row["relative_l2"] <= gates["per_parameter_gradient_relative_l2_max"])
        rows.append(row)
    return rows


def adamw_audit(G, active, before, after, hp, learning_rate, observed_norm, observed_clip, gates):
    """FP64 clip + AdamW formula driven by the candidate's own gradient (and generated coefficient)."""
    from megakernel_lab.tinymoe.optimizer import EXPERT
    squares = 0.0
    flags = {}
    for name, g in G.items():
        m = EXPERT.match(name)
        flags[name] = not m or bool(active[f"{int(m[1])}"][int(m[2])] > 0)
        if flags[name]:
            squares += float(g.double().square().sum())
    norm = math.sqrt(squares)
    coefficient = min(1.0, hp["clip_norm"] / (norm + 1e-6))
    norm_error = abs(observed_norm - norm) / max(norm, 1e-20)
    coefficient_error = abs(observed_clip - coefficient) / max(coefficient, 1e-20)
    b1, b2, eps, wd = hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"]
    rows = []
    for name, p in before["parameters"].items():
        opt0, opt1 = before["optimizer_state"], after["optimizer_state"]
        m0, v0, t0 = (opt0[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
        m1, v1, t1 = (opt1[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
        if not flags[name]:
            passed = all(bitwise(x, y) for x, y in ((p, after["parameters"][name]), (m0, m1), (v0, v1), (t0, t1)))
            rows.append(dict(name=name, inactive=True, passed=passed))
            continue
        g = G[name].double() * observed_clip
        count = int(t0) + 1
        em = b1 * m0.double() + (1 - b1) * g
        ev = b2 * v0.double() + (1 - b2) * g.square()
        ep = p.double() * (1 - learning_rate * wd) if p.ndim >= 2 else p.double()
        ep = ep - learning_rate / (1 - b1 ** count) * em / (ev.sqrt() / math.sqrt(1 - b2 ** count) + eps)
        metrics = {key: distance(x, y) for key, x, y in (("parameter", after["parameters"][name], ep),
                                                         ("exp_avg", m1, em), ("exp_avg_sq", v1, ev))}
        passed = int(t1) == count and all(r["finite"] and (r["max_abs"] <= gates["own_gradient_optimizer_absolute_max"]
                   or r["relative_l2"] <= gates["own_gradient_optimizer_relative_l2_max"]) for r in metrics.values())
        rows.append(dict(name=name, inactive=False, passed=passed, metrics=metrics))
    clip = dict(norm_fp64=norm, observed_norm=observed_norm, norm_relative_error=norm_error,
                coefficient_fp64=coefficient, observed_coefficient=observed_clip, coefficient_relative_error=coefficient_error,
                passed=norm_error <= gates["clip_norm_relative_max"] and coefficient_error <= gates["clip_norm_relative_max"])
    return clip, dict(passed=all(r["passed"] for r in rows), inactive_parameters=sum(r["inactive"] for r in rows),
                      worst=max((r for r in rows if not r["inactive"]),
                                key=lambda r: max(m["relative_l2"] for m in r["metrics"].values()))["name"],
                      rows=rows)


def native(a):
    from megakernel_lab.tinymoe.optimizer import CLIP, NORM
    gates, policy_sha = policy()
    path = Path(a.observation)
    receipt = json.loads(path.with_suffix(".pt.json").read_text())
    from .protocol import digest
    if receipt["bytes"] != path.stat().st_size or receipt["sha256"] != digest(path):
        raise ValueError("observation size/hash mismatch")
    obs = torch.load(path, map_location="cpu", weights_only=True)
    before, after, routes = obs["before"], obs["after"], obs["fixed_routes"]
    ref_loss, ref_grads, ref_probs = reference(a.model_dir, before["parameters"], obs["tokens"], obs["targets"],
                                               routes, 1.0, obs["hyperparameters"]["router_coefficient"])
    grads = gradient_rows(obs["gradients"], ref_grads, gates)
    observed_probs = [after["model_state"][f"probabilities.{i}"] for i in range(len(routes))]
    routing = routing_rows(observed_probs, routes, ref_probs)
    activity = activity_rows(routes, after["model_state"])
    active = {k.split(".")[1]: v for k, v in after["model_state"].items() if k.startswith("active.")}
    clip, optimizer = adamw_audit(obs["gradients"], active, before, after, obs["hyperparameters"], obs["learning_rate"],
                                  float(after["optimizer_state"][NORM]), float(after["optimizer_state"][CLIP]), gates)
    loss_error = abs(float(obs["loss"]) - ref_loss)
    worst = max(grads, key=lambda r: r["relative_l2"])
    report = dict(schema="latch-tinymoe-smallbatch-native-audit-v1", policy_sha256=policy_sha, arm=obs["arm"],
                  update_index=obs["update_index"], observation=receipt, hyperparameters=obs["hyperparameters"],
                  learning_rate=obs["learning_rate"],
                  reference="installed Transformers 4.55.4 FP64 eager; independent expert loop at observed membership",
                  objective="CE_mean + 0.02 aux", reference_loss=ref_loss, observed_loss=float(obs["loss"]),
                  loss_absolute_error=loss_error, max_gradient_relative_l2=worst["relative_l2"],
                  worst_parameter=worst["name"], gradients=grads, routing=routing, activity=activity,
                  clip=clip, optimizer=optimizer)
    report["passed"] = (loss_error <= gates["objective_absolute_max"] and all(r["passed"] for r in grads)
                        and all(r["outside_error_interval"] == 0 for r in routing) and all(r["passed"] for r in activity)
                        and clip["passed"] and optimizer["passed"])
    finish(a.out, report, dict(max_rel_l2=worst["relative_l2"], loss_error=loss_error,
                               norm_relative_error=clip["norm_relative_error"]))


def microbatch(a):
    gates, policy_sha = policy()
    folder = Path(a.observation)
    before = load(folder / "before.pt")
    mb = load(folder / f"mb{a.microbatch:02d}.pt")
    scale = 1.0 / float(mb["ce_coefficient"])
    ref_loss, ref_grads, ref_probs = reference(a.model_dir, before["parameters"], mb["tokens"], mb["targets"],
                                               mb["selected"], float(mb["ce_coefficient"]), float(mb["router_coefficient"]))
    routing = routing_rows(mb["probabilities"], mb["selected"], ref_probs)
    activity = activity_rows(mb["selected"], mb["active"])
    loss_error = abs(float(mb["loss"]) - ref_loss)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ref_grads, out / "ref_grads.pt")
    report = dict(schema="latch-tinymoe-smallbatch-microbatch-audit-v1", policy_sha256=policy_sha, microbatch=a.microbatch,
                  objective=f"(CE_mean + 0.02 aux) x {mb['ce_coefficient']:g}", gate_normalization=scale,
                  reference_loss=ref_loss, observed_loss=float(mb["loss"]), loss_absolute_error=loss_error,
                  normalized_loss_absolute_error=loss_error * scale, routing=routing, activity=activity)
    report["passed"] = (loss_error * scale <= gates["objective_absolute_max"]
                        and all(r["outside_error_interval"] == 0 for r in routing) and all(r["passed"] for r in activity))
    finish(out, report, dict(normalized_loss_error=loss_error * scale))


def accumulate(a):
    gates, policy_sha = policy()
    folder = Path(a.observation)
    before, after = load(folder / "before.pt"), load(folder / "after.pt")
    G = after["accumulated_gradient"]
    reference_sum = {n: torch.zeros_like(G[n], dtype=torch.float64) for n in G}
    micro = []
    for k in range(a.microbatches):
        directory = Path(a.reference_pattern.format(k=k))
        ref = load(directory / "ref_grads.pt")
        for n in G:
            reference_sum[n] += ref[n]
        del ref
        value = json.loads((directory / "result.json").read_text())
        micro.append(dict(microbatch=k, passed=value["passed"], normalized_loss_error=value["normalized_loss_absolute_error"],
                          routing_outside=sum(r["outside_error_interval"] for r in value["routing"]),
                          routing_disagreements=sum(r["set_disagreements"] for r in value["routing"])))
    grads = gradient_rows(G, reference_sum, gates)
    worst = max(grads, key=lambda r: r["relative_l2"])
    seen = {k.split(".")[1]: v for k, v in after["participation"].items()}
    clip, optimizer = adamw_audit(G, seen, before, after, before["hyperparameters"], before["learning_rate"],
                                  float(after["norm"]), float(after["clip"]), gates)
    report = dict(schema="latch-tinymoe-smallbatch-accumulated-audit-v1", policy_sha256=policy_sha,
                  update_index=before["update_index"], hyperparameters=before["hyperparameters"],
                  learning_rate=before["learning_rate"],
                  accumulated_gradient=dict(max_relative_l2=worst["relative_l2"], worst_parameter=worst["name"],
                                            passed=all(r["passed"] for r in grads), rows=grads),
                  clip=clip, optimizer=optimizer, microbatches=micro)
    report["passed"] = (report["accumulated_gradient"]["passed"] and clip["passed"] and optimizer["passed"]
                        and all(r["passed"] for r in micro))
    finish(a.out, report, dict(max_rel_l2=worst["relative_l2"], norm_relative_error=clip["norm_relative_error"]))


def finish(out, report, summary):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], **summary)))
    raise SystemExit(0 if report["passed"] else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("native")
    s.add_argument("--observation", required=True)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("microbatch")
    s.add_argument("--observation", required=True)
    s.add_argument("--microbatch", type=int, required=True)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("accumulate")
    s.add_argument("--observation", required=True)
    s.add_argument("--reference-pattern", required=True)
    s.add_argument("--microbatches", type=int, default=16)
    s.add_argument("--out", required=True)
    a = p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("independent references belong on CPU workers")
    torch.set_num_threads(1)
    dict(native=native, microbatch=microbatch, accumulate=accumulate)[a.command](a)


if __name__ == "__main__":
    main()
