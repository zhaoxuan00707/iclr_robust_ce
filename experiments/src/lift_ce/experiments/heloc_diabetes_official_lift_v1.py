"""Legacy HELOC/Diabetes benchmark with official robust-CF baselines.

For training-only discovery and empirical P1 endpoints use
``heloc_diabetes_bank_protocol_v3``. This module retains the historical protocol
for baseline evaluation and explicitly requested reproductions.

The driver is deliberately sharded.  ``prepare`` freezes the common experimental
objects, ``generate`` runs one official generator, ``attack`` solves one
(method, radius, lambda) conditional inner problem, and ``aggregate`` performs
pooled-attack cross-evaluation and calibration-only selection.  Final future
models are never read by ``attack`` or selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .diabetes_original_lift_comparison import shift
from .german_credit_strict_audit import solve_wachter
from .heloc_benchmark_milestone import bootstrap_budget, transport_destination, valid
from .heloc_conditional_adversary_strengthened_v1 import (
    SEED_BASE, full_metrics, optimize, radial_full_evaluation, rms,
)
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from .run_official_proplace import FrozenFrameDataset, robustx_model_from_frozen
from ..baselines.official_adapters import generate_ellice_cf, require_proplace_license
from ..data.tabular import load_tabular_three_way
from ..counterfactuals.tabular import generate_wachter_ce
from ..counterfactuals.population_path import population_mean
from ..models.tabular import TabularMLP, train_model

TAU = 0.5
DATASETS = {
    "heloc": {
        "config": {"source": "openml", "openml_name": "heloc", "openml_version": 1,
                   "missing_values": [-7, -8, -9]},
        "identity": "FICO HELOC credit-risk data",
        "favorable": "RiskPerformance=Good (encoded class 1)",
    },
    "diabetes": {
        "config": {"source": "openml", "openml_id": 37,
                   "positive_label": "tested_negative"},
        "identity": "OpenML Diabetes dataset, data_id=37 (Pima Indians Diabetes)",
        "favorable": "tested_negative: no diabetes diagnosis (encoded class 1)",
    },
}
METHODS = ("trex", "roar", "proplace", "ellice", "lift")
LAMBDAS = tuple(np.round(np.linspace(0, 1, 21), 2))
RATIOS = (0.5, 1.0, 3.0)
RESTARTS = 5
REPORT_SEEDS = (0, 1, 2)
SEARCH = {"surrogate": "softplus", "kappa": 10.0, "unroll_K": 5,
          "adversarial_lr": 0.075, "steps": 60}
ATTACK_VERSION = "solverfix_bank_v2"
NEAR_OPTIMAL_HINGE_TOL = 0.005
FUTURE_ENVS = ("generic_population_shift", "target_side_shift", "mixed_shift")
FUTURE_SEVERITIES = (0.5, 1.0, 2.5)
FUTURE_REALIZATIONS = 5


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_model(path: Path, dim: int):
    model = TabularMLP(dim, [64, 32])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def git_version(path: Path):
    snapshot = path.parent.parent / "provenance/core_import.json"
    if path.parent.name == "vendor" and snapshot.is_file():
        vendor = json.loads(snapshot.read_text())["vendors"].get(path.name)
        return vendor["commit"] if vendor else "not-vendored"
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def versions(repo: Path):
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "lime", "ellice", "RobustX", "gurobipy"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not-installed"
    return {
        "packages": packages,
        "implementations": {
            "T-Rex": {"implementation": "RobustX STCE", "commit": git_version(repo/"vendor/RobustX"),
                       "defaults": {"K": 40, "threshold": 0.4,
                         "DeltaRobustnessEvaluator": {"delta": .005, "bias_delta": .005,
                                                       "M": 10000, "epsilon": .0001},
                         "stability_samples": 1000, "stability_sd": .1}},
            "ROAR": {"implementation": "official robust-ce-inn algorithm with CARLA defaults; compatibility import avoids CARLA TensorFlow package initializer",
                     "algorithm_commit": git_version(repo/"vendor/robust-ce-inn"),
                     "CARLA_commit": git_version(repo/"vendor/CARLA"),
                     "defaults": {"lr": .01, "lambda": .01, "delta_max": .01,
                                  "norm": 1, "t_max_min": .5, "loss_type": "BCE",
                                  "y_target": [0, 1], "binary_cat_features": True,
                                  "loss_threshold": .001, "discretize": False,
                                  "sample": True, "lime_seed": 0, "seed": 0}},
            "PROPLACE": {"implementation": "RobustX PROPLACE", "commit": git_version(repo/"vendor/RobustX"),
                         "defaults": {"delta": .005, "bias_delta": .005,
                                      "epsilon": .0001, "k": 10}},
            "ElliCE": {"implementation": "ellice continuous generator", "commit": git_version(repo/"vendor/ellice"),
                       "defaults": {"robustness_epsilon": .01, "regularization_coefficient": .0001,
                                    "total_CFs": 1, "features_to_vary": "all", "target_class": 1,
                                    "learning_rate": .1, "max_iterations": 1000, "patience": 50,
                                    "robustness_weight": 1, "proximity_weight": 0,
                                    "gradient_mode": "min-max"}},
        },
        "lift": {"lambda_grid": list(LAMBDAS), "radius_ratios": list(RATIOS),
                 "conditional_adversary_restarts": RESTARTS, **SEARCH,
                 "full_update": {"optimizer": "AdamW", "lr": 2e-4,
                                 "weight_decay": 1e-4, "epochs": 8, "batch_size": 256},
                 "near_optimal_absolute_hinge_tolerance": NEAR_OPTIMAL_HINGE_TOL,
                 "attack_pooling": "all lambda values and restarts within each radius; full-update cross-evaluation"},
    }


def seed_dir(root: Path, dataset: str, seed: int):
    return root / dataset / f"seed{seed}"


def load_common(root: Path, dataset: str, seed: int, data_home: str):
    sd = seed_dir(root, dataset, seed)
    d = load_tabular_three_way(DATASETS[dataset]["config"], data_home, .6, .2, seed)
    common = torch.load(sd/"common.pt", map_location="cpu", weights_only=True)
    model = load_model(sd/"models/current.pt", d["train_x"].shape[1])
    return sd, d, common, model


def predictive_features(x, y, count=3):
    scores = []
    for j in range(x.shape[1]):
        c = np.corrcoef(x[:, j].numpy(), y.numpy())[0, 1]
        scores.append(0 if not np.isfinite(c) else abs(float(c)))
    return torch.as_tensor(np.argsort(scores)[-count:][::-1].copy())


def make_future_models(sd, d, current, b_ref, seed, split):
    """Create disjoint calibration or final model IDs from train-side shifts."""
    model_dir = sd/"models"/f"future_{split}"
    model_dir.mkdir(parents=True, exist_ok=True)
    features = predictive_features(d["train_x"], d["train_y"])
    records = []
    offset = 40_000 if split == "calibration" else 80_000
    for ei, env in enumerate(FUTURE_ENVS):
        for severity in FUTURE_SEVERITIES:
            for k in range(FUTURE_REALIZATIONS):
                realization = seed*1_000_000 + offset + ei*10000 + int(severity*100)*10 + k
                sx, sy, kind = shift(env, d["train_x"], d["train_y"], severity, k,
                                     realization, b_ref, features)
                path = model_dir/f"{env}_s{severity:g}_k{k}.pt"
                if not path.exists():
                    fm, _ = train_model(sx, sy, d["cal_x"], d["cal_y"], [64,32],
                                        40, 1e-3, realization)
                    torch.save(fm.state_dict(), path)
                n = min(len(sx), len(d["train_x"]))
                severity_measured = float(rms(sx[:n]-d["train_x"][:n]))
                records.append({"split": split, "future_model_id": f"{split}|{env}|s{severity:g}|k{k}",
                    "environment": env, "severity_control": severity, "realization": k,
                    "seed": realization, "kind": kind, "solver_proxy_severity": severity_measured,
                    "path": str(path)})
    pd.DataFrame(records).to_csv(sd/f"future_{split}_manifest.csv", index=False)


def prepare(args):
    sd = seed_dir(args.output, args.dataset, args.seed)
    (sd/"models").mkdir(parents=True, exist_ok=True)
    d = load_tabular_three_way(DATASETS[args.dataset]["config"], args.data_home, .6, .2, args.seed)
    model_path = sd/"models/current.pt"
    if model_path.exists(): current = load_model(model_path, d["train_x"].shape[1])
    else:
        current, _ = train_model(d["train_x"], d["train_y"], d["cal_x"], d["cal_y"],
                                 [64,32], 40, 1e-3, args.seed)
        torch.save(current.state_dict(), model_path)
    with torch.no_grad():
        unfavorable = (torch.sigmoid(current(d["test_x"])) < TAU).nonzero().flatten()
    g = torch.Generator().manual_seed(args.seed + 2)
    chosen = unfavorable[torch.randperm(len(unfavorable), generator=g)[:min(128, len(unfavorable))]]
    b_ref, _ = bootstrap_budget(d["cal_x"], 20, min(128, len(d["cal_x"])), args.seed+4)
    common = {"source": d["test_x"][chosen], "source_ids": torch.as_tensor(np.asarray(d["test_ids"])[chosen.numpy()]),
              "test_indices": chosen, "b_ref": float(b_ref)}
    atomic_torch_save(common, sd/"common.pt")
    make_future_models(sd, d, current, b_ref, args.seed, "calibration")
    make_future_models(sd, d, current, b_ref, args.seed, "final")
    metadata = {"dataset": args.dataset, **DATASETS[args.dataset], "seed": args.seed,
                "split": "60/20/20", "num_sources": len(chosen), "source_ids": common["source_ids"].tolist(),
                "b_ref": b_ref, "selection_data": "calibration only", "final_test_role": "evaluation only",
                "shared_across_methods": ["split", "factual IDs", "calibration future IDs", "final future IDs"],
                "actionability": {"generation_policy": "official defaults; all features mutable",
                  "reported_audit": "immutable age and pregnancies for Diabetes; training-range violations for all datasets"},
                **versions(Path(__file__).resolve().parents[3])}
    (sd/"metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"prepared": str(sd), "sources": len(chosen), "b_ref": b_ref}, indent=2))


def robustx_objects(d, current):
    from robustx.lib.tasks.ClassificationTask import ClassificationTask
    rx_model = robustx_model_from_frozen(current, d["train_x"].shape[1])
    frame = pd.DataFrame(d["train_x"].numpy(), columns=d["feature_names"])
    frame["target"] = d["train_y"].numpy()
    data = FrozenFrameDataset(seed=0); data.data = frame
    return ClassificationTask(rx_model, data), rx_model


def roar_one(x, d, current, index, delta_max=.01, project=None):
    """CARLA ROAR equations/defaults without its TensorFlow package initializer."""
    from lime.lime_tabular import LimeTabularExplainer
    from scipy.optimize import linprog
    def predict_proba(v):
        with torch.no_grad():
            p = torch.sigmoid(current(torch.as_tensor(v, dtype=torch.float32))).numpy().reshape(-1)
        return np.column_stack((1-p, p))
    # CARLA defaults use one shared LIME seed (0), default Ridge regressor,
    # 5000 samples, continuous features, and sampling around the instance.
    np.random.seed(0)
    explainer = LimeTabularExplainer(d["train_x"].numpy(),
        training_labels=d["train_y"].numpy(), feature_names=d["feature_names"],
        discretize_continuous=False, sample_around_instance=True,
        categorical_names=[])
    exp = explainer.explain_instance(x.numpy(), predict_proba,
                                     num_features=d["train_x"].shape[1])
    coef = np.zeros(d["train_x"].shape[1], dtype=np.float32)
    for j, value in exp.local_exp[1]: coef[j] = value
    intercept = float(exp.intercept[1])
    coeff = torch.tensor(coef); bias = torch.tensor([intercept]); target = torch.tensor(1.)
    torch.manual_seed(0); candidate = x.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([candidate], lr=.01, amsgrad=True)
    previous = torch.tensor(0.); difference = .001 + 1
    started = time.perf_counter()
    while float(difference) > .001 and time.perf_counter()-started < 30.:
        old = previous.detach().clone(); w = torch.cat((coeff, bias)).requires_grad_(True)
        augmented = torch.cat((candidate, torch.ones(1)))
        worst_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(w@augmented), target)
        gw = torch.autograd.grad(worst_loss, w)[0]
        solution = linprog((-gw.detach().numpy()), bounds=[(-delta_max,delta_max)]*len(w), method="highs")
        if not solution.success: raise RuntimeError("ROAR weight-perturbation LP failed")
        dw=torch.tensor(solution.x[:-1],dtype=torch.float32);db=torch.tensor(solution.x[-1],dtype=torch.float32)
        optimizer.zero_grad(); probability=torch.sigmoid((coeff+dw)@candidate+bias+db).squeeze()
        previous=torch.nn.functional.binary_cross_entropy(probability,target)+.01*torch.dist(candidate,x,1)
        previous.backward();optimizer.step();difference=torch.dist(old,previous.detach())
        if project is not None:
            with torch.no_grad(): candidate.copy_(project(candidate))
    return candidate.detach()


def generate(args):
    sd, d, common, current = load_common(args.output, args.dataset, args.seed, args.data_home)
    source, names = common["source"], list(d["feature_names"])
    out = sd/f"{args.method}_original.pt"
    if out.exists():
        loaded = torch.load(out, map_location="cpu", weights_only=True)
        saved = {"counterfactuals": list(loaded["counterfactuals"]),
                 "native_failure": list(loaded["native_failure"].bool()),
                 "runtime_seconds": list(loaded["runtime_seconds"])}
    else: saved = {"counterfactuals": [], "native_failure": [], "runtime_seconds": []}
    if args.method == "lift":
        # LIFT has its own base CF generator. Baseline methods never receive a
        # lambda; this artifact is the sole source for the LIFT lambda path.
        # Generate from the seed-specific frozen f0 and factuals.  Earlier
        # exploratory artifacts reused seed 0 here; the WRO experiment must
        # keep splits/factuals/models matched within every replication.
        started = time.perf_counter()
        if args.dataset == "diabetes":
            zs = list(solve_wachter(current, source, .5, {}, {}, 5, 500,
                                    args.seed + 90_000)[0].detach())
        else:
            zs = list(generate_wachter_ce(current, source, .5, 300, .03, .05, 20.))
        saved = {"counterfactuals": zs, "native_failure": [False]*len(source),
                 "runtime_seconds": [time.perf_counter()-started]*len(source)}
        atomic_torch_save({**saved, "counterfactuals": torch.stack(saved["counterfactuals"]),
                           "native_failure": torch.zeros(len(source),dtype=torch.bool),
                           "runtime_seconds": torch.zeros(len(source)), "completed": len(source)}, out)
        print(f"{args.dataset} seed={args.seed} lift base CFs loaded: {len(source)}", flush=True)
        return
    task = rx = generator = None
    if args.method in ("trex", "proplace"):
        task, rx = robustx_objects(d, current)
        if args.method == "trex":
            from robustx.generators.robust_CE_methods.STCE import STCE
            generator = STCE(task)
        else:
            PROPLACE = require_proplace_license(); generator = PROPLACE(task)
    for i in range(len(saved["counterfactuals"]), len(source)):
        start = time.perf_counter(); failed = False
        try:
            factual = pd.Series(source[i].numpy(), index=names)
            if args.method in ("trex", "proplace"):
                result = generator.generate_for_instance(factual)
                if result.empty: raise RuntimeError("empty official result")
                z = torch.as_tensor(result.iloc[0].to_numpy(float), dtype=torch.float32)
            elif args.method == "ellice":
                result = generate_ellice_cf(source[i].numpy(), current, d["train_x"].numpy(),
                    d["train_y"].numpy(), names, target_class=1, robustness_epsilon=.01)
                if result.empty: raise RuntimeError("empty official result")
                z = torch.as_tensor(result[names].iloc[0].to_numpy(float), dtype=torch.float32)
            else: z = roar_one(source[i], d, current, i)
            if not torch.isfinite(z).all(): raise RuntimeError("non-finite official result")
        except Exception as exc:
            # A solver/license failure is experiment-wide infrastructure failure,
            # never a per-factual generation failure.
            if args.method == "proplace" and type(exc).__name__ == "GurobiError":
                raise
            print(f"{args.dataset} seed={args.seed} {args.method} i={i} failed: {type(exc).__name__}: {exc}", flush=True)
            z, failed = source[i].clone(), True
        saved["counterfactuals"].append(z); saved["native_failure"].append(failed)
        saved["runtime_seconds"].append(time.perf_counter()-start)
        atomic_torch_save({**saved, "counterfactuals": torch.stack(saved["counterfactuals"]),
                           "native_failure": torch.as_tensor(saved["native_failure"]),
                           "runtime_seconds": torch.as_tensor(saved["runtime_seconds"]),
                           "completed": i+1}, out)
        # Continue with list form after each atomic checkpoint.
        saved["counterfactuals"] = list(torch.load(out, map_location="cpu", weights_only=True)["counterfactuals"])
        saved["native_failure"] = list(torch.load(out, map_location="cpu", weights_only=True)["native_failure"].bool())
        saved["runtime_seconds"] = list(torch.load(out, map_location="cpu", weights_only=True)["runtime_seconds"])
        print(f"{args.dataset} seed={args.seed} {args.method}: {i+1}/{len(source)}", flush=True)


def path_artifact(sd, d, common, current, method):
    obj = torch.load(sd/f"{method}_original.pt", map_location="cpu", weights_only=True)
    base = obj["counterfactuals"]
    cached = sd/f"{method}_lift_path.pt"
    if cached.exists():
        frozen = torch.load(cached, map_location="cpu", weights_only=True)
        if not torch.equal(base, frozen["base"]):
            raise ValueError("CF population changed: use a new deployment instance directory")
        return base, frozen["destination"], frozen["path"], obj
    favorable = d["train_x"][valid(current, d["train_x"])]
    destination = transport_destination(base, favorable)
    path = {f"{lam:.2f}": ((1-lam)*base + lam*destination) for lam in LAMBDAS}
    atomic_torch_save({"base": base, "destination": destination, "path": path,
                       "native_failure": obj["native_failure"],
                       "generation_runtime_seconds": obj["runtime_seconds"]}, sd/f"{method}_lift_path.pt")
    return base, destination, path, obj


def attack_instance_id(d, common, current, path):
    """Fingerprint the frozen data, classifier, path and full-update code."""
    digest = hashlib.sha256(inspect.getsource(fine_tune).encode())
    tensors = {"train_x": d["train_x"], "train_y": d["train_y"],
               **{f"model/{k}": v for k, v in current.state_dict().items()},
               **{f"path/{k}": v for k, v in path.items()}}
    digest.update(str(float(common["b_ref"])).encode())
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def attack(args):
    sd, d, common, current = load_common(args.output, args.dataset, args.seed, args.data_home)
    _, _, path, _ = path_artifact(sd, d, common, current, args.method)
    instance_id = attack_instance_id(d, common, current, path)
    search_model = copy.deepcopy(current).to(args.device).train()
    radius_indices = range(len(RATIOS)) if args.all_grid else (args.radius_index,)
    lambda_indices = range(len(LAMBDAS)) if args.all_grid else (args.lambda_index,)
    for ri in radius_indices:
      for li in lambda_indices:
        ratio, lam = RATIOS[ri], LAMBDAS[li]
        radius, q = ratio*float(common["b_ref"]), path[f"{lam:.2f}"].detach()
        od = sd/"attacks"/args.method/f"r{ratio:g}"/f"lambda{lam:.2f}"; od.mkdir(parents=True, exist_ok=True)
        for restart in range(RESTARTS):
            # A strengthened solver appends a new bank; legacy attacks survive.
            target = od/f"restart{restart}_{ATTACK_VERSION}.pt"
            if target.exists():
                existing = torch.load(target, map_location="cpu", weights_only=False)
                if existing.get("instance_id") != instance_id:
                    raise ValueError(f"deployment instance changed: {target}")
                continue
            seed = SEED_BASE + args.seed*10_000_000 + ri*100_000 + li*1000 + restart
            delta, trace, runtime, peak = optimize(search_model, d["train_x"].to(args.device),
                d["train_y"].to(args.device), q.to(args.device), radius, seed,
                SEARCH["surrogate"], SEARCH["kappa"], SEARCH["unroll_K"],
                SEARCH["adversarial_lr"], SEARCH["steps"])
            candidates = []
            def retain_candidate(metrics, candidate):
                candidates.append({"delta": candidate.detach().cpu().clone(),
                                   "source_full_metrics": metrics})
            winner, radial = radial_full_evaluation(current, d["train_x"], d["train_y"], q,
                delta, radius, seed+700_000, args.device, candidate_sink=retain_candidate)
            row, selected_delta, _ = winner
            atomic_torch_save({"delta": selected_delta, "search_seed": seed,
                               "source_lambda": lam, "radius_ratio": ratio,
                               "full_update_seed": seed+700_000,
                               "attack_version": ATTACK_VERSION, "candidates": candidates,
                               "instance_id": instance_id,
                               "source_full_metrics": row, "runtime_seconds": runtime}, target)
            trace.to_csv(target.with_name(target.stem+"_trace.csv"), index=False)
            radial.to_csv(target.with_name(target.stem+"_radial.csv"), index=False)
            print(f"attack {args.dataset} s{args.seed} {args.method} r={ratio} lambda={lam} restart={restart}", flush=True)


def actionability(source, z, dataset, names):
    lo, hi = source.new_tensor([]), source.new_tensor([])
    # Bounds are added by aggregate using training data; this helper only audits
    # semantically immutable raw features where names are available.
    immutable = {"age", "preg", "pregnancies"} if dataset == "diabetes" else set()
    idx = [i for i,n in enumerate(names) if str(n).lower() in immutable]
    return ((z[:, idx]-source[:, idx]).abs() > 1e-5).any(1) if idx else torch.zeros(len(z), dtype=torch.bool)


def metric_row(dataset, seed, method, point, ratio, lam, q, source, initial_valid,
               native_failure, current, models, cost, action_violation, runtime):
    success = ~native_failure
    cur = (torch.sigmoid(current(q)) >= TAU) & success
    vals = []
    for fm in models: vals.append(float((((torch.sigmoid(fm(q)) >= TAU) & success).float().mean())))
    subset = success & initial_valid
    subset_vals = []
    for fm in models:
        good = torch.sigmoid(fm(q)) >= TAU
        subset_vals.append(float(good[subset].float().mean()) if subset.any() else np.nan)
    return {"dataset": dataset, "seed": seed, "baseline": method, "operating_point": point,
        "r_over_b_ref": ratio, "lambda": lam, "generation_success": float(success.float().mean()),
        "generation_failures": int((~success).sum()), "current_validity": float(cur.float().mean()),
        "mean_future_validity": float(np.mean(vals)), "worst_future_validity": float(np.min(vals)),
        "recourse_cost": cost, "actionability_violation_rate": float(action_violation.float().mean()),
        "runtime_seconds": runtime, "initially_valid_count": int(subset.sum()),
        "conditional_mean_future_validity": float(np.nanmean(subset_vals)),
        "conditional_worst_future_validity": float(np.nanmin(subset_vals))}


@torch.no_grad()
def evaluate_attack_bank(attacks, q, radius, weights=None):
    """Evaluate each full-update model on one Q; maximize AFTER averaging.

    This finite feasible bank is a lower approximation, not a WRO certificate.
    Source lambda records discovery provenance and never filters evaluation.
    """
    evaluations = []
    for source_lam, restart, obj, fm in attacks:
        distance = float(rms(obj["delta"]))
        if not np.isfinite(distance) or distance > radius + max(1e-7, radius*1e-5):
            raise ValueError("attack is not feasible for the evaluation radius")
        pred = torch.sigmoid(fm(q))
        if not torch.isfinite(pred).all():
            raise ValueError("nonfinite full-update predictions")
        metrics = obj["source_full_metrics"]
        evaluations.append({
            "attack_source_lambda": source_lam, "restart": restart,
            "attack_id": obj.get("attack_id", str(restart)),
            "attack_version": obj.get("attack_version", "legacy"),
            "solver_budget_distance": distance, "budget_fraction": distance/radius,
            "ot_style_distance": float(metrics.get("ot_style_distance", np.nan)),
            "true_hinge": float(population_mean(torch.relu(TAU-pred), weights)),
            "invalidity": float(population_mean((pred<TAU).float(), weights)),
            "validity": float(population_mean((pred>=TAU).float(), weights)),
            "mean_score": float(population_mean(pred, weights)),
        })
    if not evaluations:
        raise ValueError("empty attack bank")
    return evaluations, max(evaluations, key=lambda x: (x["true_hinge"], x["invalidity"]))


def aggregate(args):
    sd, d, common, current = load_common(args.output, args.dataset, args.seed, args.data_home)
    base, destination, path, original = path_artifact(sd, d, common, current, args.method)
    source = common["source"]; native_failure = original["native_failure"].bool()
    initial_valid = torch.sigmoid(current(base)) >= TAU
    # Baselines are evaluated at their own original operating point.  They do
    # not receive LIFT's lambda-conditional WRO attacks or radius selection.
    if args.method != "lift":
        manifest = pd.read_csv(sd/"future_final_manifest.csv")
        final_models = [load_model(Path(p), d["train_x"].shape[1]) for p in manifest.path]
        names = list(d["feature_names"]); train_lo=d["train_x"].min(0).values; train_hi=d["train_x"].max(0).values
        q = base; semantic = actionability(source, q, args.dataset, names)
        bounds=((q<train_lo-1e-5)|(q>train_hi+1e-5)).any(1); violations=semantic|bounds
        row = metric_row(args.dataset,args.seed,args.method,"lambda_0",np.nan,0.,q,source,
                         initial_valid,native_failure,current,final_models,
                         float(torch.linalg.vector_norm(q-source,dim=1).mean()),violations,
                         float(original["runtime_seconds"].sum()))
        pd.DataFrame([row]).to_csv(sd/f"{args.method}_final_results.csv",index=False)
        print(pd.DataFrame([row]).to_string(index=False))
        return
    curve_rows, conditional_rows = [], []
    instance_id = attack_instance_id(d, common, current, path)
    selected = {}
    for ratio in RATIOS:
        attacks = []
        for source_lam in LAMBDAS:
            ad = sd/"attacks"/args.method/f"r{ratio:g}"/f"lambda{source_lam:.2f}"
            for restart in range(RESTARTS):
                # These files are produced locally by this experiment and also
                # contain NumPy scalar diagnostics. PyTorch's restricted tensor-
                # only loader rejects those harmless metadata scalars.
                files = sorted(ad.glob(f"restart{restart}*.pt"))
                if not files:
                    raise RuntimeError(f"missing attacks: {ad}, restart={restart}")
                # Update_full depends on P_delta, not on the target Q.  Construct
                # it once, then cross-evaluate that identical deployment model
                # over every lambda in the radius pool.
                for attack_file in files:
                    saved = torch.load(attack_file, map_location="cpu", weights_only=False)
                    if saved.get("instance_id", instance_id) != instance_id:
                        raise ValueError(f"deployment instance mismatch: {attack_file}")
                    for ci, candidate in enumerate(saved.get("candidates", [saved])):
                        obj = {**saved, **candidate,
                               "attack_id": f"{attack_file.relative_to(sd)}:{ci}"}
                        delta = obj["delta"]
                        radius = ratio*float(common["b_ref"])
                        if (delta.shape != d["train_x"].shape
                                or not torch.isfinite(delta).all()
                                or float(rms(delta)) > radius + max(1e-7, radius*1e-5)
                                or not np.isclose(obj["radius_ratio"], ratio)):
                            raise ValueError(f"incompatible or infeasible attack: {attack_file}")
                        # Preserve legacy Update seeds, and replay new candidates
                        # with the exact seed used in their radial full evaluation.
                        fm = fine_tune(current, d["train_x"]+delta, d["train_y"],
                            obj.get("full_update_seed", obj["search_seed"]+900_000), args.device)
                        attacks.append((source_lam, restart, obj, fm))
        for lam in LAMBDAS:
            q = path[f"{lam:.2f}"]
            evaluations, worst = evaluate_attack_bank(attacks, q, ratio*float(common["b_ref"]))
            for record in evaluations:
                conditional_rows.append({"dataset": args.dataset, "seed": args.seed,
                    "baseline": args.method, "r_over_b_ref": ratio, "target_lambda": lam, **record})
            own = [e["true_hinge"] for e in evaluations if np.isclose(e["attack_source_lambda"], lam)]
            legacy = [e["true_hinge"] for e in evaluations if e["attack_version"] == "legacy"]
            legacy_risk = max(legacy) if legacy else np.nan
            cost = float(torch.linalg.vector_norm(q-source, dim=1).mean())
            curve_rows.append({"dataset": args.dataset, "seed": args.seed, "baseline": args.method,
                "r_over_b_ref": ratio, "lambda": lam, "pooled_worst_hinge": worst["true_hinge"],
                "pooled_worst_invalidity": worst["invalidity"], "pooled_worst_validity": 1-worst["invalidity"],
                "recourse_cost": cost, "winning_attack_lambda": worst["attack_source_lambda"],
                "winning_restart": worst["restart"],
                "winning_attack_id": worst["attack_id"], "bank_size": len(evaluations),
                "own_lambda_worst_hinge": max(own),
                "cross_lambda_risk_increase": worst["true_hinge"]-max(own),
                "legacy_bank_hinge": legacy_risk,
                "bank_strengthening_risk_increase": worst["true_hinge"]-legacy_risk,
                "winning_budget_fraction": worst["budget_fraction"],
                "winning_solver_budget_distance": worst["solver_budget_distance"],
                "winning_ot_style_distance": worst["ot_style_distance"]})
        rc = pd.DataFrame([x for x in curve_rows if x["r_over_b_ref"] == ratio])
        best_h = rc.pooled_worst_hinge.min()
        robust = rc[np.isclose(rc.pooled_worst_hinge, best_h, atol=1e-6)].sort_values("lambda").iloc[0]
        selected[(ratio,"robustness_only")] = robust
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(sd/f"{args.method}_conditional_robustness_cost_curves.csv", index=False)
    # Retain the historical filename, now containing the full source/target
    # cross-product. Ranking uses max_attack(mean_CF(true_hinge)).
    pd.DataFrame(conditional_rows).to_csv(sd/f"{args.method}_conditional_attack_evaluation.csv", index=False)
    # Diagnostic curves: cost is plotted/reported but never enters selection.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ratio in RATIOS:
        rc = curves[curves.r_over_b_ref == ratio]
        chosen_lam = float(selected[(ratio, "robustness_only")]["lambda"])
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(rc["lambda"], rc["pooled_worst_hinge"], marker="o")
        ax.axvline(chosen_lam, color="tab:red", linestyle="--", label=f"selected λ={chosen_lam:g}")
        ax.set(xlabel="lambda", ylabel="Phi-hat (pooled worst true hinge)")
        ax.legend(); fig.tight_layout(); fig.savefig(sd/f"{args.method}_phi_r{ratio:g}.png", dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(rc["lambda"], rc["recourse_cost"], marker="o")
        ax.set(xlabel="lambda", ylabel="mean recourse cost"); fig.tight_layout()
        fig.savefig(sd/f"{args.method}_cost_r{ratio:g}.png", dpi=150); plt.close(fig)
    manifest = pd.read_csv(sd/"future_calibration_manifest.csv")
    cal_models = [load_model(Path(p), d["train_x"].shape[1]) for p in manifest.path]
    candidates=[]
    for (ratio, kind), row in selected.items():
        q=path[f"{row['lambda']:.2f}"]; vals=[float((torch.sigmoid(m(q))>=TAU).float().mean()) for m in cal_models]
        candidates.append({"selector":kind,"r_over_b_ref":ratio,"lambda":float(row["lambda"]),
            "train_pooled_hinge":float(row.pooled_worst_hinge),"cal_mean":np.mean(vals),"cal_worst":np.min(vals),
            "cost":float(row.recourse_cost)})
    cand=pd.DataFrame(candidates); chosen=[]
    for kind,g in cand.groupby("selector"):
        chosen.append(g.sort_values(["cal_worst","cal_mean","cost","r_over_b_ref"],
                      ascending=[False,False,True,True]).iloc[0])
    chosen=pd.DataFrame(chosen); cand["selected_radius"]=False
    for r in chosen.itertuples(): cand.loc[(cand.selector==r.selector)&(cand.r_over_b_ref==r.r_over_b_ref),"selected_radius"]=True
    cand.to_csv(sd/f"{args.method}_calibration_selection.csv",index=False)
    final_manifest=pd.read_csv(sd/"future_final_manifest.csv")
    final_models=[load_model(Path(p),d["train_x"].shape[1]) for p in final_manifest.path]
    rows=[]; names=list(d["feature_names"]); train_lo=d["train_x"].min(0).values;train_hi=d["train_x"].max(0).values
    points=[("lambda_0",np.nan,0.),("lambda_1",np.nan,1.)]
    points += [(r.selector,float(r.r_over_b_ref),float(r.lambda_)) for r in chosen.rename(columns={"lambda":"lambda_"}).itertuples()]
    for point, ratio, lam in points:
        q=path[f"{lam:.2f}"]; semantic=actionability(source,q,args.dataset,names)
        bounds=((q<train_lo-1e-5)|(q>train_hi+1e-5)).any(1); violations=semantic|bounds
        cost=float(torch.linalg.vector_norm(q-source,dim=1).mean())
        runtime=float(original["runtime_seconds"].sum())
        rows.append(metric_row(args.dataset,args.seed,args.method,point,ratio,lam,q,source,
            initial_valid,native_failure,current,final_models,cost,violations,runtime))
    pd.DataFrame(rows).to_csv(sd/f"{args.method}_final_results.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def report(args):
    records=[]
    for dataset in DATASETS:
        for seed in REPORT_SEEDS:
            for method in METHODS:
                path=seed_dir(args.output,dataset,seed)/f"{method}_final_results.csv"
                if path.exists(): records.append(pd.read_csv(path))
    if not records: raise RuntimeError("No completed final result shards")
    raw=pd.concat(records,ignore_index=True)
    # Lambda is a LIFT-CF parameter only. Baselines contribute their original
    # CF row; their exploratory lambda/cost-aware rows are excluded.
    raw = raw[((raw.baseline == "lift") & (raw.operating_point != "cost_aware_near_optimal")) |
              ((raw.baseline != "lift") & (raw.operating_point == "lambda_0"))].copy()
    raw.to_csv(args.output/"all_seed_results.csv",index=False)
    metrics=["generation_success","current_validity","mean_future_validity","worst_future_validity",
             "recourse_cost","actionability_violation_rate","runtime_seconds",
             "conditional_mean_future_validity","conditional_worst_future_validity"]
    summary=raw.groupby(["dataset","baseline","operating_point"])[metrics].agg(["mean","std","count"])
    summary.columns=[f"{a}_{b}" for a,b in summary.columns];summary.reset_index().to_csv(args.output/"summary_with_uncertainty.csv",index=False)
    pd.DataFrame([
        {"dataset": ds, "baseline": "proplace", "status": "blocked",
         "reason": "official RobustX PROPLACE MILP exceeds the installed size-limited Gurobi license; no valid official run"}
        for ds in DATASETS
    ]).to_csv(args.output/"proplace_status.csv", index=False)
    originals=raw[raw.operating_point=="lambda_0"]
    paired=raw.merge(originals,on=["dataset","seed","baseline"],suffixes=("","_original"))
    for metric in metrics: paired[f"paired_improvement__{metric}"]=paired[metric]-paired[f"{metric}_original"]
    paired.to_csv(args.output/"paired_improvements.csv",index=False)
    print(summary.reset_index().to_string(index=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=("prepare","generate","attack","aggregate","report"))
    ap.add_argument("--dataset",choices=DATASETS);ap.add_argument("--method",choices=METHODS)
    ap.add_argument("--seed",type=int,default=0);ap.add_argument("--radius-index",type=int,default=0)
    ap.add_argument("--lambda-index",type=int,default=0);ap.add_argument("--data-home",default="/home/xu.zhao/robust_ce_data")
    ap.add_argument("--all-grid",action="store_true",help="run all radius/lambda shards in one process")
    ap.add_argument("--legacy-protocol", action="store_true",
                    help="explicitly reproduce the historical test-CF/barycentric protocol")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase!="report" and args.dataset is None:ap.error("--dataset is required")
    if args.phase in ("generate","attack","aggregate") and args.method is None:ap.error("--method is required")
    if args.method == "lift" and args.phase in ("attack", "aggregate") and not args.legacy_protocol:
        ap.error("Use lift_ce.experiments.heloc_diabetes_bank_protocol_v3 for the corrected protocol; "
                 "--legacy-protocol is only for historical reproduction")
    {"prepare":prepare,"generate":generate,"attack":attack,"aggregate":aggregate,"report":report}[args.phase](args)


if __name__ == "__main__": main()
