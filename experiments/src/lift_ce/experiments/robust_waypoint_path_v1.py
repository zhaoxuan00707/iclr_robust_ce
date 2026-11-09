"""Frozen training-derived waypoint paths; calibration witnesses are excluded."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from . import heloc_diabetes_bank_protocol_v3 as protocol
from ..counterfactuals.robust_waypoints import ModelBank, fit_waypoints
from ..counterfactuals.population_path import PopulationPath, WaypointPopulationPath, population_mean
from .full_update_adversary_v1 import differentiable_full_update
from .heloc_conditional_adversary_strengthened_v1 import rms, search_loss
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from .diabetes_original_lift_comparison import shift


def context(args):
    root, meta, train, current = protocol.load_instance(args)
    bank_dir, bank = protocol.verify_bank(root, meta, "bank_full_update_v1")
    out = root/"robust_waypoint_path_v1"
    out.mkdir(exist_ok=True)
    return root, meta, train, current, bank_dir, bank, out


def train_path(args):
    _, meta, train, current, bank_dir, bank, out = context(args)
    filename = out/f"training_fit{args.fit_seed}.pt"
    if filename.exists():
        saved = torch.load(filename, weights_only=True)
        if saved["steps"] != args.steps or saved["bank_sha256"] != protocol.file_hash(bank_dir/"manifest.json"):
            raise ValueError("existing training fit has different settings")
        return
    models = [protocol.legacy.load_model(bank_dir/a["model_file"], train["x"].shape[1]) for a in bank["attacks"]]
    model_bank = ModelBank(models, args.device)
    target = train["target"].to(args.device)
    points, objective, history = fit_waypoints(model_bank, current, target,
        train["x"].min(0).values.to(args.device), train["x"].max(0).values.to(args.device),
        radius=3., steps=args.steps, seed=args.fit_seed)
    protocol.legacy.atomic_torch_save({"waypoints": points, "training_objective": objective,
        "fit_seed": args.fit_seed, "steps": args.steps, "radius": 3., "split": "train",
        "parent_instance_id": meta["instance_id"], "bank_sha256": protocol.file_hash(bank_dir/"manifest.json")}, filename)
    pd.DataFrame(history).to_csv(filename.with_suffix(".csv"), index=False)


def freeze_path(args):
    _, meta, _, _, bank_dir, _, out = context(args)
    target = out/"frozen_path.pt"
    if target.exists():
        return
    fits = [torch.load(out/f"training_fit{i}.pt", weights_only=True) for i in range(7)]
    for fit in fits:
        if (fit["parent_instance_id"] != meta["instance_id"] or fit["split"] != "train"
                or fit["bank_sha256"] != protocol.file_hash(bank_dir/"manifest.json")):
            raise ValueError("training path provenance mismatch")
    winner = min(fits, key=lambda f: (f["training_objective"], f["fit_seed"]))
    protocol.legacy.atomic_torch_save(winner, target)
    protocol.write_json(out/"path_manifest.json", {
        "parent_instance_id": meta["instance_id"], "path_sha256": protocol.file_hash(target),
        "training_bank_sha256": winner["bank_sha256"], "fit_seed": winner["fit_seed"],
        "training_objective": winner["training_objective"], "split": "train",
        "path": "base -> learned target waypoints -> original P1; fixed coupling",
        "maximum_target_displacement": 3., "calibration_used_for_fit": False})


def load_path(out, population):
    manifest = json.loads((out/"path_manifest.json").read_text())
    if protocol.file_hash(out/"frozen_path.pt") != manifest["path_sha256"]:
        raise ValueError("learned path changed after freezing")
    fit = torch.load(out/"frozen_path.pt", weights_only=True)
    return WaypointPopulationPath(PopulationPath(**population["path"]), fit["waypoints"]), manifest


def discover(args):
    root, meta, train, current, bank_dir, bank, out = context(args)
    population = torch.load(root/"train_population.pt", weights_only=True)
    path, path_manifest = load_path(out, population)
    ratio, lam = meta["ratios"][args.radius_index], meta["lambdas"][args.lambda_index]
    (out/"attacks").mkdir(exist_ok=True)
    filename = out/"attacks"/f"waypoint_r{ratio:g}_l{lam:.2f}.pt"
    if filename.exists():
        saved = torch.load(filename, weights_only=False)
        if saved["path_sha256"] != path_manifest["path_sha256"]:
            raise ValueError("attack path changed")
        return
    members = [a for a in bank["attacks"] if a["radius_ratio"] == ratio]
    models = [protocol.legacy.load_model(bank_dir/a["model_file"], train["x"].shape[1]) for a in members]
    q, weights = path.at(lam).to(args.device), path.weights.to(args.device)
    risks = ModelBank(models, args.device).risks(q, weights=weights)
    worst = int(risks.argmax())
    attack_file, index = members[worst]["attack_id"].rsplit(":", 1)
    saved = torch.load(root/"attacks"/attack_file, weights_only=False)
    delta = saved["candidates"][int(index)]["delta"].to(args.device).clone().requires_grad_(True)
    initial_risk = float(risks[worst])
    model = copy.deepcopy(current).to(args.device).train()
    optimizer = torch.optim.Adam([delta], lr=.075)
    radius = ratio*meta["b_ref"]
    x, y = train["x"].to(args.device), train["y"].to(args.device)
    candidates, history = [], []
    for step in range(11):
        pred = differentiable_full_update(model, x, y, delta, q, meta["full_update_seed"])
        objective = search_loss(pred, "softplus", 10., weights)
        gradient = torch.autograd.grad(objective, delta)[0]
        if not (torch.isfinite(pred).all() and torch.isfinite(gradient).all()):
            raise ValueError("nonfinite new-path discovery")
        hinge = float(population_mean(torch.relu(.5-pred), weights))
        if step == 0 and abs(hinge-initial_risk) > 1e-4:
            raise ValueError("full replay and frozen warm-start model disagree")
        history.append({"iteration": step, "true_hinge": hinge,
                        "gradient_norm": float(gradient.norm()), "finite": True})
        candidates.append(delta.detach().cpu().clone())
        if step == 10:
            break
        optimizer.zero_grad()
        delta.grad = -gradient
        optimizer.step()
        with torch.no_grad():
            norm = rms(delta)
            if norm > radius:
                delta.mul_(radius/norm)
    protocol.legacy.atomic_torch_save({"parent_instance_id": meta["instance_id"],
        "path_sha256": path_manifest["path_sha256"], "discovery_split": "train",
        "source_lambda": lam, "radius_ratio": ratio, "candidates": candidates,
        "full_update_seed": meta["full_update_seed"]}, filename)
    pd.DataFrame(history).to_csv(filename.with_suffix(".csv"), index=False)
    print(f"new-path attack r={ratio:g} lambda={lam:g}: {initial_risk:.6f} -> {hinge:.6f}", flush=True)


def freeze_bank(args):
    root, meta, train, current, bank_dir, bank, out = context(args)
    _, path_manifest = load_path(out, torch.load(root/"train_population.pt", weights_only=True))
    if (out/"bank_manifest.json").exists():
        return
    records = [{**a, "model_path": str(bank_dir/a["model_file"]), "origin": "reference"} for a in bank["attacks"]]
    (out/"models").mkdir(exist_ok=True)
    for ratio in meta["ratios"]:
        for lam in (.25, .5, .75):
            file = out/"attacks"/f"waypoint_r{ratio:g}_l{lam:.2f}.pt"
            saved = torch.load(file, weights_only=False)
            if (saved["parent_instance_id"] != meta["instance_id"] or saved["discovery_split"] != "train"
                    or saved["path_sha256"] != path_manifest["path_sha256"]
                    or saved["full_update_seed"] != meta["full_update_seed"]):
                raise ValueError("new-path attack provenance mismatch")
            for index, delta in enumerate(saved["candidates"]):
                radius = ratio*meta["b_ref"]
                if (delta.shape != train["x"].shape or not torch.isfinite(delta).all()
                        or float(rms(delta)) > radius+1e-5):
                    raise ValueError("new-path attack is infeasible")
                model = fine_tune(current, train["x"]+delta, train["y"], meta["full_update_seed"], args.device)
                target = out/"models"/f"{file.stem}_candidate{index}.pt"
                protocol.legacy.atomic_torch_save(model.state_dict(), target)
                records.append({"attack_id": f"{file.name}:{index}", "radius_ratio": ratio,
                    "source_lambda": lam, "model_path": str(target), "model_sha256": protocol.file_hash(target),
                    "origin": "waypoint_training_discovery"})
    protocol.write_json(out/"bank_manifest.json", {"parent_instance_id": meta["instance_id"],
        "path_sha256": path_manifest["path_sha256"], "reference_bank_sha256": protocol.file_hash(bank_dir/"manifest.json"),
        "attacks": records, "selection_population_used": False})


def verified_bank(out):
    manifest = json.loads((out/"bank_manifest.json").read_text())
    if protocol.file_hash(out/"frozen_path.pt") != manifest["path_sha256"]:
        raise ValueError("path changed after bank freezing")
    for a in manifest["attacks"]:
        if protocol.file_hash(Path(a["model_path"])) != a["model_sha256"]:
            raise ValueError("frozen candidate model changed")
    return manifest


def verified_selection(root, out):
    selected = json.loads((out/"selection.json").read_text())
    checks = [(out/"frozen_path.pt", selected["path_sha256"]),
              (out/"bank_manifest.json", selected["bank_sha256"]),
              (root/"cal_population.pt", selected["cal_population_sha256"])]
    checks.extend((Path(p), digest) for p, digest in selected["calibration_model_hashes"].items())
    if any(protocol.file_hash(p) != digest for p, digest in checks):
        raise ValueError("frozen selection inputs changed")
    return selected


def select(args):
    root, meta, train, current, _, _, out = context(args)
    bank = verified_bank(out)
    if (out/"selection.json").exists():
        verified_selection(root, out)
        return
    if protocol.file_hash(root/"cal_population.pt") != (root/"cal_population.sha256").read_text().strip():
        raise ValueError("calibration CF population changed")
    population = torch.load(root/"cal_population.pt", weights_only=True)
    learned, _ = load_path(out, population)
    paths = {"linear_control": learned.population, "learned_waypoint": learned}
    models = [protocol.legacy.load_model(Path(a["model_path"]), train["x"].shape[1]) for a in bank["attacks"]]
    ensemble = ModelBank(models, args.device)
    cal_manifest = pd.read_csv(Path(meta["original_instance"])/"future_calibration_manifest.csv")
    cal_files = [root/"update_alignment_feasibility_v1/matched_models"/f"model{i}.pt" for i in cal_manifest.index]
    cal_models = [protocol.legacy.load_model(p, train["x"].shape[1]).to(args.device) for p in cal_files]
    current = current.to(args.device)
    curves, evaluations = [], []
    with torch.no_grad():
        for name, path in paths.items():
            for lam in meta["lambdas"]:
                q, weights = path.at(lam).to(args.device), path.weights.to(args.device)
                risks = ensemble.risks(q, weights=weights)
                validity = [float(population_mean((torch.sigmoid(m(q))>=.5).float(), weights)) for m in cal_models]
                cur = float(population_mean((torch.sigmoid(current(q))>=.5).float(), weights))
                origins = population["source"][learned.population.source_index].to(args.device)
                cost = float(population_mean(torch.linalg.vector_norm(q-origins, dim=1), weights))
                for a, risk in zip(bank["attacks"], risks.cpu().tolist()):
                    evaluations.append({"path": name, "lambda": lam, "radius_ratio": a["radius_ratio"],
                                        "attack_id": a["attack_id"], "true_hinge": risk})
                for ratio in meta["ratios"]:
                    indices = [i for i, a in enumerate(bank["attacks"]) if a["radius_ratio"] == ratio]
                    worst = max(indices, key=lambda i: float(risks[i]))
                    curves.append({"path": name, "lambda": lam, "r_over_b_ref": ratio,
                        "pooled_worst_hinge": float(risks[worst]), "winning_attack_id": bank["attacks"][worst]["attack_id"],
                        "bank_size": len(indices), "cal_worst": min(validity), "cal_mean": float(np.mean(validity)),
                        "current_validity": cur, "recourse_cost": cost})
    frame = pd.DataFrame(curves)
    selected, candidates = [], []
    for name, group in frame.groupby("path"):
        choices = []
        for _, rows in group.groupby("r_over_b_ref"):
            minimum = rows.pooled_worst_hinge.min()
            best = rows[np.isclose(rows.pooled_worst_hinge, minimum, atol=1e-6)].sort_values("lambda").iloc[0]
            choices.append(best.to_dict())
        candidates.extend(choices)
        selected.append(pd.DataFrame(choices).sort_values(["cal_worst", "cal_mean", "recourse_cost", "r_over_b_ref"],
                                                         ascending=[False, False, True, True]).iloc[0].to_dict())
    frame.to_csv(out/"calibration_curves.csv", index=False)
    pd.DataFrame(evaluations).to_csv(out/"calibration_attack_cross_evaluation.csv", index=False)
    pd.DataFrame(candidates).to_csv(out/"calibration_candidates.csv", index=False)
    protocol.write_json(out/"selection.json", {"path_sha256": protocol.file_hash(out/"frozen_path.pt"),
        "bank_sha256": protocol.file_hash(out/"bank_manifest.json"),
        "cal_population_sha256": protocol.file_hash(root/"cal_population.pt"),
        "operator": "matched_full_update", "selected": selected,
        "calibration_model_hashes": {str(p): protocol.file_hash(p) for p in cal_files}})
    print(pd.DataFrame(selected).to_string(index=False), flush=True)


def replay_final(args):
    root, meta, train, current, _, _, out = context(args)
    if not (out/"selection.json").exists():
        raise ValueError("freeze the selection before opening final environments")
    verified_selection(root, out)
    manifest = pd.read_csv(Path(meta["original_instance"])/"future_final_manifest.csv")
    features = protocol.legacy.predictive_features(train["x"], train["y"])
    (out/"aligned_final_models").mkdir(exist_ok=True)
    for index, row in manifest.iterrows():
        if index % args.workers != args.worker:
            continue
        target = out/"aligned_final_models"/f"model{index}.pt"
        if target.exists():
            continue
        x, y, _ = shift(row.environment, train["x"], train["y"], row.severity_control,
                        int(row.realization), int(row.seed), meta["b_ref"], features)
        model = fine_tune(current, x, y, meta["full_update_seed"], args.device)
        protocol.legacy.atomic_torch_save(model.state_dict(), target)


def evaluate(args):
    root, meta, train, current, _, _, out = context(args)
    selection = verified_selection(root, out)
    verified_bank(out)
    population = protocol.holdout_population(root, meta, train, current, "test")
    learned, _ = load_path(out, population)
    paths = {"linear_control": learned.population, "learned_waypoint": learned}
    manifest = pd.read_csv(Path(meta["original_instance"])/"future_final_manifest.csv")
    rows = []
    with torch.no_grad():
        for operator in ("aligned_full_update", "scratch_stress"):
            files = ([out/"aligned_final_models"/f"model{i}.pt" for i in manifest.index]
                     if operator == "aligned_full_update" else list(map(Path, manifest.path)))
            models = [protocol.legacy.load_model(p, train["x"].shape[1]) for p in files]
            points = [(s["path"], s["lambda"], s["r_over_b_ref"]) for s in selection["selected"]]
            points.extend([("endpoint_0", 0., np.nan), ("endpoint_1", 1., np.nan)])
            for name, lam, ratio in points:
                path = paths.get(name, learned.population)
                q, weights = path.at(lam), path.weights
                validity = [float(population_mean((torch.sigmoid(m(q))>=.5).float(), weights)) for m in models]
                origins = population["source"][learned.population.source_index]
                rows.append({"operator": operator, "path": name, "lambda": lam, "r_over_b_ref": ratio,
                    "current_validity": float(population_mean((torch.sigmoid(current(q))>=.5).float(), weights)),
                    "mean_future_validity": float(np.mean(validity)), "worst_future_validity": min(validity),
                    "recourse_cost": float(population_mean(torch.linalg.vector_norm(q-origins, dim=1), weights))})
    pd.DataFrame(rows).to_csv(out/"final_results.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "freeze_path", "discover", "freeze_bank", "select", "replay_final", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="heloc")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fit-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radius-index", type=int, default=1)
    parser.add_argument("--lambda-index", type=int, default=10)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {"train": train_path, "freeze_path": freeze_path, "discover": discover,
     "freeze_bank": freeze_bank, "select": select, "replay_final": replay_final,
     "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
