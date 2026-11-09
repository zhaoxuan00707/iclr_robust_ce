"""Joint selection over a frozen finite union of training-derived 1D paths."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import robust_waypoint_path_v1 as previous
from . import heloc_diabetes_bank_protocol_v3 as protocol
from .full_update_adversary_v1 import differentiable_full_update
from .heloc_conditional_adversary_strengthened_v1 import rms, search_loss
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from ..counterfactuals.population_path import PopulationPath, WaypointPopulationPath, population_mean
from ..counterfactuals.robust_waypoints import ModelBank


def context(args):
    root, meta, train, current = protocol.load_instance(args)
    old = root / "robust_waypoint_path_v1"
    out = root / "finite_waypoint_paths_v2"
    out.mkdir(exist_ok=True)
    return root, meta, train, current, old, out


def freeze_paths(args):
    root, meta, train, _, old, out = context(args)
    if (out / "paths_manifest.json").exists():
        load_paths(out, torch.load(root / "train_population.pt", weights_only=True))
        return
    bank = previous.verified_bank(old)
    training_bank = root / "banks/bank_full_update_v1/manifest.json"
    entries = [{"k": 0, "name": "straight", "type": "linear"}]
    for seed in range(7):
        source = old / f"training_fit{seed}.pt"
        fit = torch.load(source, weights_only=True)
        if (fit["parent_instance_id"] != meta["instance_id"] or fit["split"] != "train"
                or fit["bank_sha256"] != protocol.file_hash(training_bank)):
            raise ValueError("training fit provenance mismatch")
        wp = fit["waypoints"]
        if (wp.shape != train["target"].shape or not torch.isfinite(wp).all()
                or (wp < train["x"].min(0).values-1e-5).any()
                or (wp > train["x"].max(0).values+1e-5).any()
                or torch.linalg.vector_norm(wp-train["target"], dim=1).max() > fit["radius"]+1e-5):
            raise ValueError("invalid waypoint support")
        target = out / f"path{seed+1}.pt"
        protocol.legacy.atomic_torch_save(fit, target)
        entries.append({"k": seed+1, "name": f"waypoint_seed{seed}", "type": "waypoint",
                        "file": target.name, "sha256": protocol.file_hash(target),
                        "source_sha256": protocol.file_hash(source), "fit_seed": seed,
                        "training_objective": fit["training_objective"]})
    protocol.write_json(out / "paths_manifest.json", {
        "parent_instance_id": meta["instance_id"], "split": "train", "paths": entries,
        "reference_bank_sha256": protocol.file_hash(old / "bank_manifest.json"),
        "reference_model_count": len(bank["attacks"]),
        "fit_policy": "retain all seven existing training-only fits; no calibration filtering",
        "source_lambdas": [.25, .5, .75], "search_steps": 10,
        "selection": "joint minimum true bank hinge over k and lambda per radius; no cost constraint"})


def load_paths(out, population):
    manifest = json.loads((out / "paths_manifest.json").read_text())
    base = PopulationPath(**population["path"])
    paths = {0: base}
    for entry in manifest["paths"][1:]:
        path = out / entry["file"]
        if protocol.file_hash(path) != entry["sha256"]:
            raise ValueError("frozen path changed")
        paths[entry["k"]] = WaypointPopulationPath(base, torch.load(path, weights_only=True)["waypoints"])
    return paths, manifest


def old_bank(old, out):
    manifest = json.loads((out / "paths_manifest.json").read_text())
    if protocol.file_hash(old / "bank_manifest.json") != manifest["reference_bank_sha256"]:
        raise ValueError("reference bank changed")
    return previous.verified_bank(old)


def discover(args):
    root, meta, train, current, old, out = context(args)
    paths, path_manifest = load_paths(out, torch.load(root / "train_population.pt", weights_only=True))
    bank = old_bank(old, out)
    ratio, lam = meta["ratios"][args.radius_index], meta["lambdas"][args.lambda_index]
    (out / "attacks").mkdir(exist_ok=True)
    filename = out / "attacks" / f"path{args.path_index}_r{ratio:g}_l{lam:.2f}.pt"
    path_hash = protocol.file_hash(out / "paths_manifest.json")
    if filename.exists():
        saved = torch.load(filename, weights_only=False)
        if saved["paths_sha256"] != path_hash or saved["parent_instance_id"] != meta["instance_id"]:
            raise ValueError("existing discovery inputs changed")
        return
    # A smaller-radius feasible attack is also feasible at a larger radius.
    members = [a for a in bank["attacks"] if a["radius_ratio"] <= ratio]
    models = [protocol.legacy.load_model(Path(a["model_path"]), train["x"].shape[1]) for a in members]
    path = paths[args.path_index]
    q, weights = path.at(lam).to(args.device), path.weights.to(args.device)
    risks = ModelBank(models, args.device).risks(q, weights=weights)
    worst = int(risks.argmax())
    record = members[worst]
    attack_file, index = record["attack_id"].rsplit(":", 1)
    attack_root = old if record["origin"] == "waypoint_training_discovery" else root
    saved = torch.load(attack_root / "attacks" / attack_file, weights_only=False)
    candidate = saved["candidates"][int(index)]
    initial = candidate["delta"] if isinstance(candidate, dict) else candidate
    delta = initial.to(args.device).clone().requires_grad_(True)
    initial_risk = float(risks[worst])
    del models, risks
    model = copy.deepcopy(current).to(args.device).train()
    optimizer = torch.optim.Adam([delta], lr=.075)
    radius = ratio*meta["b_ref"]
    x, y = train["x"].to(args.device), train["y"].to(args.device)
    candidates, history = [], []
    for step in range(path_manifest["search_steps"]+1):
        pred = differentiable_full_update(model, x, y, delta, q, meta["full_update_seed"])
        gradient = torch.autograd.grad(search_loss(pred, "softplus", 10., weights), delta)[0]
        if not (torch.isfinite(pred).all() and torch.isfinite(gradient).all()):
            raise ValueError("nonfinite path discovery")
        hinge = float(population_mean(torch.relu(.5-pred), weights))
        if step == 0 and abs(hinge-initial_risk) > 1e-4:
            raise ValueError("warm start replay mismatch")
        history.append({"iteration": step, "true_hinge": hinge,
                        "gradient_norm": float(gradient.norm()), "finite": True})
        candidates.append(delta.detach().cpu().clone())
        if step == path_manifest["search_steps"]:
            break
        optimizer.zero_grad()
        delta.grad = -gradient
        optimizer.step()
        with torch.no_grad():
            norm = rms(delta)
            if norm > radius:
                delta.mul_(radius/norm)
    protocol.legacy.atomic_torch_save({"parent_instance_id": meta["instance_id"],
        "paths_sha256": path_hash, "discovery_split": "train", "source_path": args.path_index,
        "source_lambda": lam, "radius_ratio": ratio, "candidates": candidates,
        "warm_start_attack": record["attack_id"], "full_update_seed": meta["full_update_seed"]}, filename)
    pd.DataFrame(history).to_csv(filename.with_suffix(".csv"), index=False)
    print(f"k={args.path_index} r={ratio:g} lambda={lam:g}: {initial_risk:.6f} -> {hinge:.6f}", flush=True)


def materialize(args):
    root, meta, train, current, _, out = context(args)
    _, manifest = load_paths(out, torch.load(root / "train_population.pt", weights_only=True))
    jobs = [(e["k"], r, l) for e in manifest["paths"] for r in meta["ratios"] for l in manifest["source_lambdas"]]
    (out / "models").mkdir(exist_ok=True)
    for number, (k, ratio, lam) in enumerate(jobs):
        if number % args.workers != args.worker:
            continue
        file = out / "attacks" / f"path{k}_r{ratio:g}_l{lam:.2f}.pt"
        saved = torch.load(file, weights_only=False)
        if (saved["parent_instance_id"] != meta["instance_id"] or saved["discovery_split"] != "train"
                or saved["paths_sha256"] != protocol.file_hash(out / "paths_manifest.json")
                or saved["full_update_seed"] != meta["full_update_seed"]
                or saved["source_path"] != k or saved["radius_ratio"] != ratio or saved["source_lambda"] != lam
                or len(saved["candidates"]) != manifest["search_steps"]+1):
            raise ValueError("attack provenance mismatch")
        records = []
        for index, delta in enumerate(saved["candidates"]):
            if (delta.shape != train["x"].shape or not torch.isfinite(delta).all()
                    or float(rms(delta)) > ratio*meta["b_ref"]+1e-5):
                raise ValueError("infeasible attack")
            target = out / "models" / f"{file.stem}_candidate{index}.pt"
            # A job sidecar is published only after every official update finishes.
            model = fine_tune(current, train["x"]+delta, train["y"], meta["full_update_seed"], args.device)
            protocol.legacy.atomic_torch_save(model.state_dict(), target)
            records.append({"attack_id": f"finite_v2/{file.name}:{index}", "radius_ratio": ratio,
                "source_path": k, "source_lambda": lam, "model_path": str(target),
                "model_sha256": protocol.file_hash(target), "origin": "finite_path_training_discovery"})
        protocol.write_json(file.with_suffix(".models.json"), {
            "attack_sha256": protocol.file_hash(file), "attacks": records})
        print(f"materialized {file.stem}", flush=True)


def freeze_bank(args):
    root, meta, _, _, old, out = context(args)
    _, manifest = load_paths(out, torch.load(root / "train_population.pt", weights_only=True))
    if (out / "bank_manifest.json").exists():
        verified_bank(out)
        return
    records = list(old_bank(old, out)["attacks"])
    for entry in manifest["paths"]:
        for ratio in meta["ratios"]:
            for lam in manifest["source_lambdas"]:
                file = out / "attacks" / f'path{entry["k"]}_r{ratio:g}_l{lam:.2f}.pt'
                saved = json.loads(file.with_suffix(".models.json").read_text())
                if protocol.file_hash(file) != saved["attack_sha256"]:
                    raise ValueError("materialized attack changed")
                records.extend(saved["attacks"])
    if len({a["attack_id"] for a in records}) != len(records):
        raise ValueError("duplicate attack identifiers")
    protocol.write_json(out / "bank_manifest.json", {"parent_instance_id": meta["instance_id"],
        "paths_sha256": protocol.file_hash(out / "paths_manifest.json"),
        "reference_bank_sha256": protocol.file_hash(old / "bank_manifest.json"),
        "attacks": records, "selection_population_used": False,
        "radius_compatibility": "include all attacks discovered at radius <= evaluation radius"})
    verified_bank(out)


def verified_bank(out):
    bank = json.loads((out / "bank_manifest.json").read_text())
    if protocol.file_hash(out / "paths_manifest.json") != bank["paths_sha256"]:
        raise ValueError("path collection changed")
    for a in bank["attacks"]:
        if protocol.file_hash(Path(a["model_path"])) != a["model_sha256"]:
            raise ValueError("frozen model changed")
    return bank


def joint_choices(frame):
    """Exact finite-grid risk minima per radius; cost breaks exact risk ties."""
    choices = []
    for _, rows in frame.groupby("r_over_b_ref", sort=True):
        best = rows.sort_values(["pooled_worst_hinge", "recourse_cost", "k", "lambda"]).iloc[0]
        choices.append(best.to_dict())
    return pd.DataFrame(choices)


def choose_radius(candidates):
    return candidates.sort_values(["cal_worst", "cal_mean", "recourse_cost", "r_over_b_ref"],
                                  ascending=[False, False, True, True]).iloc[0].to_dict()


def select(args):
    root, meta, train, current, _, out = context(args)
    bank = verified_bank(out)
    if (out / "selection.json").exists():
        verified_selection(root, out)
        return
    if protocol.file_hash(root / "cal_population.pt") != (root / "cal_population.sha256").read_text().strip():
        raise ValueError("calibration population changed")
    population = torch.load(root / "cal_population.pt", weights_only=True)
    paths, manifest = load_paths(out, population)
    models = [protocol.legacy.load_model(Path(a["model_path"]), train["x"].shape[1]) for a in bank["attacks"]]
    ensemble = ModelBank(models, args.device)
    cal_manifest = pd.read_csv(Path(meta["original_instance"]) / "future_calibration_manifest.csv")
    cal_files = [root / "update_alignment_feasibility_v1/matched_models" / f"model{i}.pt" for i in cal_manifest.index]
    cal_models = [protocol.legacy.load_model(p, train["x"].shape[1]).to(args.device) for p in cal_files]
    current = current.to(args.device)
    reference_count = manifest["reference_model_count"]
    curves, evaluations = [], []
    with torch.no_grad():
        for k, path in paths.items():
            for lam in meta["lambdas"]:
                q, weights = path.at(lam).to(args.device), path.weights.to(args.device)
                risks = ensemble.risks(q, weights=weights)
                validity = [float(population_mean((torch.sigmoid(m(q)) >= .5).float(), weights)) for m in cal_models]
                cur = float(population_mean((torch.sigmoid(current(q)) >= .5).float(), weights))
                origins = population["source"][paths[0].source_index].to(args.device)
                cost = float(population_mean(torch.linalg.vector_norm(q-origins, dim=1), weights))
                for a, risk in zip(bank["attacks"], risks.cpu().tolist()):
                    evaluations.append({"k": k, "lambda": lam, "discovery_radius": a["radius_ratio"],
                                        "attack_id": a["attack_id"], "true_hinge": risk})
                for ratio in meta["ratios"]:
                    indices = [i for i, a in enumerate(bank["attacks"]) if a["radius_ratio"] <= ratio]
                    worst = max(indices, key=lambda i: float(risks[i]))
                    old_risk = max(float(risks[i]) for i in indices if i < reference_count)
                    curves.append({"k": k, "lambda": lam, "r_over_b_ref": ratio,
                        "pooled_worst_hinge": float(risks[worst]), "reference_bank_hinge": old_risk,
                        "risk_increase": float(risks[worst])-old_risk,
                        "winning_attack_id": bank["attacks"][worst]["attack_id"], "bank_size": len(indices),
                        "cal_worst": min(validity), "cal_mean": float(np.mean(validity)),
                        "current_validity": cur, "recourse_cost": cost})
            print(f"calibrated path {k}", flush=True)
    frame = pd.DataFrame(curves)
    candidates = joint_choices(frame)
    control_candidates = joint_choices(frame[frame.k == 0])
    selected = [{**choose_radius(candidates), "method": "finite_union"},
                {**choose_radius(control_candidates), "method": "straight_control"}]
    frame.to_csv(out / "calibration_curves.csv", index=False)
    pd.DataFrame(evaluations).to_csv(out / "calibration_attack_cross_evaluation.csv", index=False)
    candidates.to_csv(out / "joint_candidates.csv", index=False)
    control_candidates.to_csv(out / "control_candidates.csv", index=False)
    protocol.write_json(out / "selection.json", {
        "paths_sha256": protocol.file_hash(out / "paths_manifest.json"),
        "bank_sha256": protocol.file_hash(out / "bank_manifest.json"),
        "cal_population_sha256": protocol.file_hash(root / "cal_population.pt"),
        "calibration_model_hashes": {str(p): protocol.file_hash(p) for p in cal_files},
        "operator": "matched_full_update", "selected": selected,
        "cost_budget": None, "selection_rule": "joint risk minimum per radius, then aligned calibration radius selection"})
    print(pd.DataFrame(selected).to_string(index=False), flush=True)


def verified_selection(root, out):
    selection = json.loads((out / "selection.json").read_text())
    checks = [(out / "paths_manifest.json", selection["paths_sha256"]),
              (out / "bank_manifest.json", selection["bank_sha256"]),
              (root / "cal_population.pt", selection["cal_population_sha256"])]
    checks.extend((Path(p), h) for p, h in selection["calibration_model_hashes"].items())
    if any(protocol.file_hash(p) != h for p, h in checks):
        raise ValueError("selection inputs changed")
    return selection


def evaluate(args):
    root, meta, train, current, old, out = context(args)
    selection = verified_selection(root, out)
    verified_bank(out)
    population = protocol.holdout_population(root, meta, train, current, "test")
    paths, _ = load_paths(out, population)
    # Reuse the already computed exact deployment updates; never open them in select.
    manifest = pd.read_csv(Path(meta["original_instance"]) / "future_final_manifest.csv")
    rows, details = [], []
    points = selection["selected"] + [{"method": "endpoint_0", "k": 0, "lambda": 0., "r_over_b_ref": np.nan},
                                     {"method": "endpoint_1", "k": 0, "lambda": 1., "r_over_b_ref": np.nan}]
    hashes = {}
    with torch.no_grad():
        for operator in ("aligned_full_update", "scratch_stress"):
            files = ([old / "aligned_final_models" / f"model{i}.pt" for i in manifest.index]
                     if operator == "aligned_full_update" else list(map(Path, manifest.path)))
            models = [protocol.legacy.load_model(p, train["x"].shape[1]).to(args.device) for p in files]
            hashes.update({str(p): protocol.file_hash(p) for p in files})
            for selected in points:
                path = paths[int(selected["k"])]
                q, weights = path.at(selected["lambda"]).to(args.device), path.weights.to(args.device)
                validity = [float(population_mean((torch.sigmoid(m(q)) >= .5).float(), weights)) for m in models]
                origins = population["source"][paths[0].source_index].to(args.device)
                rows.append({"operator": operator, "method": selected["method"], "k": int(selected["k"]),
                    "lambda": selected["lambda"], "r_over_b_ref": selected["r_over_b_ref"],
                    "current_validity": float(population_mean((torch.sigmoid(current.to(args.device)(q)) >= .5).float(), weights)),
                    "mean_future_validity": float(np.mean(validity)), "worst_future_validity": min(validity),
                    "recourse_cost": float(population_mean(torch.linalg.vector_norm(q-origins, dim=1), weights))})
                details.extend({"operator": operator, "method": selected["method"], "model_index": i,
                                "validity": value} for i, value in enumerate(validity))
    pd.DataFrame(rows).to_csv(out / "final_results.csv", index=False)
    pd.DataFrame(details).to_csv(out / "final_per_model.csv", index=False)
    protocol.write_json(out / "evaluation_manifest.json", {"selection_sha256": protocol.file_hash(out / "selection.json"),
        "test_population_sha256": protocol.file_hash(root / "test_population.pt"), "final_model_hashes": hashes})
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze_paths", "discover", "materialize", "freeze_bank", "select", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="heloc")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--path-index", type=int, default=0)
    parser.add_argument("--radius-index", type=int, default=1)
    parser.add_argument("--lambda-index", type=int, default=10)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
