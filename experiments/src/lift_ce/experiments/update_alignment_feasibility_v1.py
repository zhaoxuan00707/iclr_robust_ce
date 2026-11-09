"""Calibration-only operator alignment and oracle feasibility diagnostics.

No final model is opened, and no diagnostic witness becomes a production CF,
attack-bank member, or lambda/r selection input.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call, stack_module_state, vmap

from . import heloc_diabetes_bank_protocol_v3 as protocol
from .diabetes_original_lift_comparison import shift
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from .heloc_conditional_adversary_strengthened_v1 import rms
from ..counterfactuals.population_path import PopulationPath, population_mean


class FrozenEnsemble:
    """Vectorized logits [models, points], differentiable only in the points."""

    def __init__(self, models, device):
        copies = [copy.deepcopy(m).to(device).eval().requires_grad_(False) for m in models]
        self.params, self.buffers = stack_module_state(copies)
        self.base = copies[0]

    def __call__(self, points):
        def forward(params, buffers, x):
            return functional_call(self.base, (params, buffers), (x,))
        return vmap(forward, in_dims=(0, 0, None))(self.params, self.buffers, points)


def project_witness(points, source, budget, lower, upper):
    """Feasible box then ball projection; source must belong to the box."""
    if ((source < lower) | (source > upper)).any():
        raise ValueError("the diagnostic box must contain the factuals")
    bounded = points.clamp(lower, upper)
    delta = bounded-source
    length = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-12)
    return source+delta*torch.minimum(torch.ones_like(length), budget/length)[:, None]


def find_joint_witnesses(ensemble, initial, source, budget, lower, upper, steps=400):
    """Maximize pointwise worst-model margin solely to diagnose feasibility.

    This deliberately answers a different question from the population WRO
    objective. A positive returned margin is a witness; a negative value is
    not an infeasibility certificate.
    """
    points = project_witness(initial, source, budget, lower, upper).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([points], lr=.05)
    best, best_margin = points.detach().clone(), torch.full_like(budget, -torch.inf)
    for step in range(steps+1):
        logits = ensemble(points)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite feasibility predictions")
        minimum = logits.min(0).values.detach()
        improved = minimum > best_margin
        best[improved] = points.detach()[improved]
        best_margin = torch.maximum(best_margin, minimum)
        if step == steps:
            break
        temperature = .5-.4*step/max(steps, 1)
        objective = -temperature*torch.logsumexp(-logits/temperature, dim=0)
        optimizer.zero_grad()
        (-objective.sum()).backward()
        optimizer.step()
        with torch.no_grad():
            points.copy_(project_witness(points, source, budget, lower, upper))
    return best, best_margin


def context(args):
    root, meta, train, current = protocol.load_instance(args)
    out = root/"update_alignment_feasibility_v1"
    out.mkdir(exist_ok=True)
    manifest = pd.read_csv(Path(meta["original_instance"])/"future_calibration_manifest.csv")
    if not manifest.split.eq("calibration").all():
        raise ValueError("this diagnostic accepts calibration shifts only")
    return root, meta, train, current, out, manifest


def replay(args):
    _, meta, train, current, out, manifest = context(args)
    (out/"matched_models").mkdir(exist_ok=True)
    features = protocol.legacy.predictive_features(train["x"], train["y"])
    for index, row in manifest.iterrows():
        if index % args.workers != args.worker:
            continue
        target = out/"matched_models"/f"model{index}.pt"
        if target.exists():
            continue
        x, y, _ = shift(row.environment, train["x"], train["y"], row.severity_control,
                        int(row.realization), int(row.seed), meta["b_ref"], features)
        model = fine_tune(current, x, y, meta["full_update_seed"], args.device)
        protocol.legacy.atomic_torch_save(model.state_dict(), target)
        print(f"matched calibration model {index}", flush=True)


def profile(args):
    root, meta, train, current, out, manifest = context(args)
    population = torch.load(root/"cal_population.pt", weights_only=True)
    path = PopulationPath(**population["path"])
    features = protocol.legacy.predictive_features(train["x"], train["y"])
    records, shift_records, shifted_targets = [], [], []
    with torch.no_grad():
        for index, row in manifest.iterrows():
            x, y, _ = shift(row.environment, train["x"], train["y"], row.severity_control,
                            int(row.realization), int(row.seed), meta["b_ref"], features)
            n = min(len(x), len(train["x"]))
            distance = float(rms(x[:n]-train["x"][:n]))
            if not np.isclose(distance, row.solver_proxy_severity, rtol=1e-5, atol=1e-6):
                raise ValueError("shift replay does not match the original manifest")
            same_labels = x.shape == train["x"].shape and torch.equal(y, train["y"])
            shift_records.append({"model_id": row.future_model_id, "kind": row.kind,
                "nominal_ratio": row.severity_control, "measured_identity_ratio": distance/meta["b_ref"],
                "same_sample_count_and_row_labels": same_labels,
                **{f"compatible_r{r:g}": same_labels and distance <= r*meta["b_ref"]+1e-5 for r in meta["ratios"]}})
            # The sampled shifted positives provide independent starts/bounds
            # for diagnostic searches, never attack discovery or selection.
            positives = x[y.bool()]
            chosen = torch.randperm(len(positives), generator=torch.Generator().manual_seed(500+index))[:64]
            shifted_targets.append(positives[chosen])
            for operator, filename in (("scratch", Path(row.path)),
                                       ("matched_full_update", out/"matched_models"/f"model{index}.pt")):
                model = protocol.legacy.load_model(filename, train["x"].shape[1])
                for lam in meta["lambdas"]:
                    pred = torch.sigmoid(model(path.at(lam)))
                    records.append({"operator": operator, "model_id": row.future_model_id,
                        "lambda": lam, "validity": float(population_mean((pred>=.5).float(), path.weights)),
                        "hinge": float(population_mean(torch.relu(.5-pred), path.weights))})
    pd.DataFrame(records).to_csv(out/"operator_calibration_evaluation.csv", index=False)
    pd.DataFrame(records).groupby(["operator", "lambda"]).agg(
        worst_validity=("validity", "min"), mean_validity=("validity", "mean"),
        worst_hinge=("hinge", "max")).to_csv(out/"operator_calibration_frontier.csv")
    pd.DataFrame(shift_records).to_csv(out/"uncertainty_compatibility.csv", index=False)
    protocol.legacy.atomic_torch_save(torch.cat(shifted_targets), out/"shifted_positive_starts.pt")
    protocol.write_json(out/"provenance.json", {"instance_id": meta["instance_id"],
        "split": "calibration", "full_update_seed": meta["full_update_seed"],
        "original_manifest_sha256": protocol.file_hash(Path(meta["original_instance"])/"future_calibration_manifest.csv"),
        "diagnostic_only": True, "final_data_used": False})


def feasibility(args):
    root, meta, train, current, out, manifest = context(args)
    population = torch.load(root/"cal_population.pt", weights_only=True)
    shifted = torch.load(out/"shifted_positive_starts.pt", weights_only=True)
    model_paths = (list(map(Path, manifest.path)) if args.operator == "scratch"
                   else [out/"matched_models"/f"model{i}.pt" for i in manifest.index])
    models = [current]+[protocol.legacy.load_model(p, train["x"].shape[1]) for p in model_paths]
    ensemble = FrozenEnsemble(models, args.device)
    count = min(32, len(population["source"]))
    source = population["source"][:count]
    base = population["path"]["base"][:count]
    bounds = torch.cat([train["x"], source] + ([shifted] if args.domain == "expanded" else []))
    lower, upper = bounds.min(0).values.to(args.device), bounds.max(0).values.to(args.device)
    budgets = (1., 3., 5., 10.)
    g = torch.Generator().manual_seed(8201)
    initials = torch.stack([base, train["target"][torch.randint(len(train["target"]), (count,), generator=g)],
                           shifted[torch.randint(len(shifted), (count,), generator=g)], source], dim=1)
    initial = initials.repeat(len(budgets), 1, 1).reshape(-1, source.shape[1]).to(args.device)
    origins = source[:, None].expand(-1, 4, -1).repeat(len(budgets), 1, 1).reshape_as(initial).to(args.device)
    limits = torch.tensor(budgets, device=args.device).repeat_interleave(count*4)
    points, margins = find_joint_witnesses(ensemble, initial, origins, limits, lower, upper, args.steps)
    # Independently validate every claimed witness with the individual models.
    with torch.no_grad():
        verified = torch.stack([m.to(args.device)(points) for m in models]).min(0).values
    torch.testing.assert_close(margins, verified, atol=2e-4, rtol=2e-4)
    costs = torch.linalg.vector_norm(points-origins, dim=1)
    feasible = (costs <= limits+1e-5) & ((points>=lower-1e-5)&(points<=upper+1e-5)).all(1)
    success = (verified >= 1e-4) & feasible
    records = []
    for bi, budget in enumerate(budgets):
        block = slice(bi*count*4, (bi+1)*count*4)
        found = success[block].view(count, 4).any(1)
        records.append({"operator": args.operator, "domain": args.domain, "budget_l2": budget,
            "factuals": count, "starts_per_factual": 4, "joint_success_count": int(found.sum()),
            "joint_success_fraction": float(found.float().mean()),
            "best_worst_model_probability": float(torch.sigmoid(verified[block].max())),
            "status": "witnesses_found" if found.any() else "none_found_not_a_certificate"})
    stem = f"joint_{args.operator}_{args.domain}"
    pd.DataFrame(records).to_csv(out/f"{stem}.csv", index=False)
    protocol.legacy.atomic_torch_save({"points": points.cpu(), "origins": origins.cpu(),
        "budgets": limits.cpu(), "verified_minimum_logit": verified.cpu(), "success": success.cpu(),
        "lower": lower.cpu(), "upper": upper.cpu(), "diagnostic_only": True}, out/f"{stem}.pt")
    print(pd.DataFrame(records).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("replay", "profile", "feasibility"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=protocol.legacy.DATASETS, default="heloc")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--operator", choices=("scratch", "matched_full_update"), default="scratch")
    parser.add_argument("--domain", choices=("observed", "expanded"), default="observed")
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers or args.steps < 0:
        parser.error("invalid worker allocation or iteration count")
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
