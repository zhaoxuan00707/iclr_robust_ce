"""Training discovery -> immutable full-update bank -> calibration -> test.

This protocol deliberately uses a new output directory. Legacy attacks targeted
test CFs and the barycentric path, and cannot enter this training-only bank.
The frozen classifier and preprocessing are reused from an existing instance.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import heloc_diabetes_official_lift_v1 as legacy
from .heloc_conditional_adversary_strengthened_v1 import optimize, radial_full_evaluation, rms
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from ..counterfactuals.population_path import PopulationPath, empirical_ot_path, population_mean

PROTOCOL = "training_bank_empirical_ot_v3"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def root_for(args):
    return legacy.seed_dir(args.output, args.dataset, args.seed)


def checked_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("use only letters, digits, underscores and hyphens")
    return value


def make_population(current, x, ids, target, dataset, seed, count):
    """Generate CFs only from the split supplied by the calling phase."""
    with torch.no_grad():
        eligible = (torch.sigmoid(current(x)) < legacy.TAU).nonzero().flatten()
    if not len(eligible):
        raise ValueError("no unfavorable factuals in this population")
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(seed))
    chosen = eligible[order[:count]]
    source = x[chosen]
    if dataset == "diabetes":
        base = legacy.solve_wachter(current, source, .5, {}, {}, 5, 500, seed+90_000)[0].detach()
    else:
        base = legacy.generate_wachter_ce(current, source, .5, 300, .03, .05, 20.).detach()
    path = empirical_ot_path(base, target)
    return {"source": source, "source_ids": ids[chosen], "path": path.state_dict()}


def prepare(args):
    root = root_for(args)
    original = legacy.seed_dir(args.instance_root, args.dataset, args.seed).resolve()
    if root.resolve() == original:
        raise ValueError("the corrected protocol requires a separate output directory")
    root.mkdir(parents=True, exist_ok=True)
    if (root/"instance.json").exists():
        load_instance(args)
        return
    if any(root.iterdir()):
        raise ValueError("incomplete or unrelated output directory; use a new empty directory")
    _, d, common, current = legacy.load_common(args.instance_root, args.dataset, args.seed, args.data_home)
    split_ids = {split: torch.as_tensor(np.asarray(d[f"{split}_ids"]).astype(np.int64))
                 for split in ("train", "cal", "test")}
    for a, b in (("train", "cal"), ("train", "test"), ("cal", "test")):
        if set(split_ids[a].tolist()) & set(split_ids[b].tolist()):
            raise ValueError("training, calibration and test IDs must be disjoint")
    with torch.no_grad():
        target = d["train_x"][torch.sigmoid(current(d["train_x"])) >= legacy.TAU]
    population = make_population(current, d["train_x"], split_ids["train"], target,
                                 args.dataset, args.seed+2, args.count)
    legacy.atomic_torch_save(current.state_dict(), root/"current.pt")
    legacy.atomic_torch_save({"x": d["train_x"], "y": d["train_y"], "ids": split_ids["train"],
                             "target": target}, root/"train.pt")
    legacy.atomic_torch_save(population, root/"train_population.pt")
    # Holdout arrays are preserved under the original preprocessing, but no
    # holdout CFs or attack objectives are constructed during discovery.
    for split in ("cal", "test"):
        legacy.atomic_torch_save({"x": d[f"{split}_x"], "ids": split_ids[split]}, root/f"{split}.pt")
    files = {p.name: file_hash(p) for p in sorted(root.glob("*.pt"))}
    metadata = {"protocol": PROTOCOL, "dataset": args.dataset, "seed": args.seed,
        "original_instance": str(original), "files": files,
        "b_ref": float(common["b_ref"]), "count": args.count,
        "lambdas": list(legacy.LAMBDAS), "ratios": list(legacy.RATIOS), "restarts": legacy.RESTARTS,
        "full_update_seed": args.seed+700_000,
        "full_update_code": hashlib.sha256(inspect.getsource(fine_tune).encode()).hexdigest(),
        "feature_names": list(d["feature_names"]),
        "target_definition": "uniform empirical training rows with f0(x) >= 0.5",
        "path": "squared-Euclidean balanced OT coupling displacement; uniform endpoint marginals",
        "uncertainty_set": "fixed-label sample displacements with RMS norm <= r*b_ref",
        "certificate": False}
    metadata["instance_id"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    write_json(root/"instance.json", metadata)


def load_instance(args):
    root = root_for(args)
    metadata = json.loads((root/"instance.json").read_text())
    if metadata["protocol"] != PROTOCOL:
        raise ValueError("wrong protocol")
    if metadata["full_update_code"] != hashlib.sha256(inspect.getsource(fine_tune).encode()).hexdigest():
        raise ValueError("full Update changed; use a new instance")
    # Discovery opens only training/model artifacts. Holdout hashes are checked
    # when their own phase actually consumes them.
    for name in ("current.pt", "train.pt", "train_population.pt"):
        if file_hash(root/name) != metadata["files"][name]:
            raise ValueError(f"frozen instance changed: {name}")
    train = torch.load(root/"train.pt", weights_only=True)
    current = legacy.load_model(root/"current.pt", train["x"].shape[1])
    return root, metadata, train, current


def discover(args):
    root, meta, train, current = load_instance(args)
    population = torch.load(root/"train_population.pt", weights_only=True)
    path = PopulationPath(**population["path"])
    model = copy.deepcopy(current).to(args.device).train()
    output = root/"attacks"
    output.mkdir(exist_ok=True)
    settings = {**legacy.SEARCH, "steps": args.steps}
    ratios = meta["ratios"] if args.all_grid else [meta["ratios"][args.radius_index]]
    lambdas = meta["lambdas"] if args.all_grid else [meta["lambdas"][args.lambda_index]]
    for ratio in ratios:
        radius = ratio*meta["b_ref"]
        for lam in lambdas:
            for restart in range(meta["restarts"]):
                target = output/f"{args.generation}_r{ratio:g}_l{lam:.2f}_restart{restart}.pt"
                if target.exists():
                    old = torch.load(target, weights_only=False)
                    if old["instance_id"] != meta["instance_id"] or old["search"] != settings:
                        raise ValueError("changed search settings require a new generation name")
                    continue
                seed = (legacy.SEED_BASE + args.seed*10_000_000
                        + meta["ratios"].index(ratio)*100_000 + meta["lambdas"].index(lam)*1000 + restart)
                q = path.at(lam)
                delta, trace, runtime, _ = optimize(model, train["x"].to(args.device),
                    train["y"].to(args.device), q.to(args.device), radius, seed,
                    settings["surrogate"], settings["kappa"], settings["unroll_K"],
                    settings["adversarial_lr"], settings["steps"], weights=path.weights)
                candidates = []
                radial_full_evaluation(current, train["x"], train["y"], q, delta, radius,
                    meta["full_update_seed"], args.device, weights=path.weights,
                    candidate_sink=lambda metrics, candidate: candidates.append({
                        "delta": candidate.detach().cpu().clone(), "source_full_metrics": metrics}))
                legacy.atomic_torch_save({"instance_id": meta["instance_id"], "discovery_split": "train",
                    "source_lambda": lam, "radius_ratio": ratio, "restart": restart,
                    "generation": args.generation, "search": settings, "runtime_seconds": runtime,
                    "full_update_seed": meta["full_update_seed"], "candidates": candidates}, target)
                trace.to_csv(target.with_suffix(".csv"), index=False)
                print(f"discovered r={ratio:g}, lambda={lam:g}, restart={restart}", flush=True)


def verify_bank(root, meta, bank_name):
    bank_dir = root/"banks"/bank_name
    manifest = json.loads((bank_dir/"manifest.json").read_text())
    if manifest["instance_id"] != meta["instance_id"]:
        raise ValueError("bank belongs to another deployment instance")
    for record in manifest["attacks"]:
        if file_hash(bank_dir/record["model_file"]) != record["model_sha256"]:
            raise ValueError("frozen full-update model changed")
    return bank_dir, manifest


def freeze(args):
    root, meta, train, current = load_instance(args)
    bank_dir = root/"banks"/args.bank_name
    if (bank_dir/"manifest.json").exists():
        verify_bank(root, meta, args.bank_name)
        return
    reference_dir, reference_models = None, {}
    if getattr(args, "reference_bank", None):
        reference_dir, reference = verify_bank(root, meta, args.reference_bank)
        reference_models = {row["attack_id"]: row for row in reference["attacks"]}
    attacks = []
    seen = set()
    for path in sorted((root/"attacks").glob("*.pt")):
        obj = torch.load(path, weights_only=False)
        if (obj.get("instance_id") != meta["instance_id"] or obj.get("discovery_split") != "train"
                or obj.get("full_update_seed") != meta["full_update_seed"]):
            raise ValueError(f"incompatible or non-training attack: {path}")
        seen.add((obj["radius_ratio"], obj["source_lambda"], obj["restart"]))
        attacks.append((path, obj))
    required = {(r, lam, k) for r in meta["ratios"] for lam in meta["lambdas"] for k in range(meta["restarts"])}
    if not required <= seen:
        raise ValueError(f"incomplete discovery grid: {len(required-seen)} shards missing")
    records = []
    bank_dir.mkdir(parents=True, exist_ok=True)
    for path, obj in attacks:
        if not obj["candidates"]:
            raise ValueError(f"empty discovery shard: {path}")
        for i, candidate in enumerate(obj["candidates"]):
            delta = candidate["delta"]
            radius = obj["radius_ratio"]*meta["b_ref"]
            if (delta.shape != train["x"].shape or not torch.isfinite(delta).all()
                    or float(rms(delta)) > radius+max(1e-7, radius*1e-5)):
                raise ValueError(f"infeasible attack: {path}:{i}")
            filename = f"{path.stem}_candidate{i}.pt"
            previous = reference_models.get(f"{path.name}:{i}")
            if previous is not None:
                if previous["discovery_sha256"] != file_hash(path):
                    raise ValueError("reference discovery artifact changed")
                shutil.copyfile(reference_dir/previous["model_file"], bank_dir/filename)
            else:
                model = fine_tune(current, train["x"]+delta, train["y"], meta["full_update_seed"], args.device)
                legacy.atomic_torch_save(model.state_dict(), bank_dir/filename)
            records.append({"attack_id": f"{path.name}:{i}", "source_lambda": obj["source_lambda"],
                "radius_ratio": obj["radius_ratio"], "generation": obj["generation"],
                "model_file": filename, "model_sha256": file_hash(bank_dir/filename),
                "discovery_sha256": file_hash(path), "budget_fraction": float(rms(delta))/radius})
    # The manifest is the completion marker, written only after every full
    # model is frozen. Calibration cannot read a partial bank.
    write_json(bank_dir/"manifest.json", {"instance_id": meta["instance_id"],
        "discovery_split": "train", "full_update_seed": meta["full_update_seed"],
        "attacks": records, "estimate": "finite-bank lower approximation"})


def holdout_population(root, meta, train, current, split):
    if file_hash(root/f"{split}.pt") != meta["files"][f"{split}.pt"]:
        raise ValueError(f"frozen {split} data changed")
    saved = root/f"{split}_population.pt"
    if saved.exists():
        if file_hash(saved) != saved.with_suffix(".sha256").read_text().strip():
            raise ValueError(f"frozen {split} population changed")
        return torch.load(saved, weights_only=True)
    data = torch.load(root/f"{split}.pt", weights_only=True)
    if split == "test":
        # Preserve the original benchmark factuals and base CFs so baseline
        # comparisons keep their paired test population.
        original = Path(meta["original_instance"])
        common = torch.load(original/"common.pt", weights_only=True)
        base = torch.load(original/"lift_original.pt", weights_only=True)["counterfactuals"]
        indices = common["test_indices"]
        old_model = legacy.load_model(original/"models/current.pt", train["x"].shape[1])
        same_model = all(torch.equal(v, old_model.state_dict()[k]) for k, v in current.state_dict().items())
        if (not same_model or not torch.equal(data["ids"][indices], common["source_ids"])
                or not torch.equal(data["x"][indices], common["source"])):
            raise ValueError("original test instance changed")
        population = {"source": common["source"], "source_ids": common["source_ids"],
                      "path": empirical_ot_path(base, train["target"]).state_dict()}
    else:
        population = make_population(current, data["x"], data["ids"], train["target"],
                                     meta["dataset"], meta["seed"]+12, meta["count"])
    legacy.atomic_torch_save(population, saved)
    saved.with_suffix(".sha256").write_text(file_hash(saved))
    return population


@torch.no_grad()
def bank_curves(bank_dir, manifest, population, ratios, lambdas, dim):
    """Stream models across ALL lambdas; each risk is a population expectation."""
    path = PopulationPath(**population["path"])
    records = []
    for attack in manifest["attacks"]:
        model = legacy.load_model(bank_dir/attack["model_file"], dim)
        for lam in lambdas:
            pred = torch.sigmoid(model(path.at(lam)))
            if not torch.isfinite(pred).all():
                raise ValueError("nonfinite full-update prediction")
            records.append({**attack, "lambda": lam,
                "true_hinge": float(population_mean(torch.relu(legacy.TAU-pred), path.weights)),
                "validity": float(population_mean((pred >= legacy.TAU).float(), path.weights))})
    evaluations = pd.DataFrame(records)
    curves = []
    for ratio in ratios:
        for lam in lambdas:
            rows = evaluations[(evaluations.radius_ratio == ratio) & (evaluations["lambda"] == lam)]
            if rows.empty:
                raise ValueError("empty radius bank")
            worst = rows.sort_values(["true_hinge", "attack_id"], ascending=[False, True]).iloc[0]
            q = path.at(lam)
            cost = population_mean(torch.linalg.vector_norm(q-population["source"][path.source_index], dim=1), path.weights)
            curves.append({"r_over_b_ref": ratio, "lambda": lam, "pooled_worst_hinge": worst.true_hinge,
                "winning_attack_id": worst.attack_id, "bank_size": len(rows), "recourse_cost": float(cost)})
    return pd.DataFrame(curves), evaluations


def select(args):
    root, meta, train, current = load_instance(args)
    bank_dir, manifest = verify_bank(root, meta, args.bank_name)
    selected_file = bank_dir/"selection.json"
    if selected_file.exists():
        selected = json.loads(selected_file.read_text())
        if (selected["bank_sha256"] != file_hash(bank_dir/"manifest.json")
                or selected["cal_population_sha256"] != file_hash(root/"cal_population.pt")):
            raise ValueError("selection bank or calibration population changed")
        return
    population = holdout_population(root, meta, train, current, "cal")
    curves, evaluations = bank_curves(bank_dir, manifest, population, meta["ratios"], meta["lambdas"], train["x"].shape[1])
    # Keep the existing radius selector: calibration-shift worst/mean validity,
    # then cost and radius. Lambda ranking now uses the frozen bank on Q_cal.
    cal_manifest = pd.read_csv(Path(meta["original_instance"])/"future_calibration_manifest.csv")
    models = [legacy.load_model(Path(p), train["x"].shape[1]) for p in cal_manifest.path]
    path = PopulationPath(**population["path"])
    choices = []
    with torch.no_grad():
        for ratio, rows in curves.groupby("r_over_b_ref"):
            minimum = rows.pooled_worst_hinge.min()
            best = rows[np.isclose(rows.pooled_worst_hinge, minimum, atol=1e-6)].sort_values("lambda").iloc[0]
            q = path.at(float(best["lambda"]))
            validity = [float(population_mean((torch.sigmoid(m(q)) >= legacy.TAU).float(), path.weights)) for m in models]
            choices.append({**best.to_dict(), "cal_worst": min(validity), "cal_mean": float(np.mean(validity))})
    candidates = pd.DataFrame(choices).sort_values(["cal_worst", "cal_mean", "recourse_cost", "r_over_b_ref"],
                                                 ascending=[False, False, True, True])
    curves.to_csv(bank_dir/"calibration_curves.csv", index=False)
    evaluations.to_csv(bank_dir/"calibration_attack_evaluation.csv", index=False)
    evaluations.groupby(["generation", "radius_ratio", "lambda"], as_index=False).true_hinge.max().to_csv(
        bank_dir/"calibration_generation_risks.csv", index=False)
    candidates.to_csv(bank_dir/"calibration_selection.csv", index=False)
    write_json(selected_file, {"instance_id": meta["instance_id"],
        "bank_sha256": file_hash(bank_dir/"manifest.json"),
        "cal_population_sha256": file_hash(root/"cal_population.pt"),
        "selection_population": "cal", **candidates.iloc[0].to_dict()})


def evaluate(args):
    root, meta, train, current = load_instance(args)
    bank_dir, _ = verify_bank(root, meta, args.bank_name)
    selection = json.loads((bank_dir/"selection.json").read_text())
    if (selection["bank_sha256"] != file_hash(bank_dir/"manifest.json")
            or selection["cal_population_sha256"] != file_hash(root/"cal_population.pt")):
        raise ValueError("bank or calibration population changed after selection")
    population = holdout_population(root, meta, train, current, "test")
    path = PopulationPath(**population["path"])
    final = pd.read_csv(Path(meta["original_instance"])/"future_final_manifest.csv")
    models = [legacy.load_model(Path(p), train["x"].shape[1]) for p in final.path]
    rows = []
    with torch.no_grad():
        for name, lam in (("lambda_0", 0.), ("lambda_1", 1.), ("robustness_only", selection["lambda"])):
            q = path.at(lam)
            validity = [float(population_mean((torch.sigmoid(m(q)) >= legacy.TAU).float(), path.weights)) for m in models]
            rows.append({"dataset": args.dataset, "seed": args.seed, "baseline": "lift",
                "operating_point": name, "lambda": lam, "r_over_b_ref": selection["r_over_b_ref"],
                "current_validity": float(population_mean((torch.sigmoid(current(q)) >= legacy.TAU).float(), path.weights)),
                "mean_future_validity": float(np.mean(validity)), "worst_future_validity": min(validity),
                "recourse_cost": float(population_mean(torch.linalg.vector_norm(q-population["source"][path.source_index], dim=1), path.weights)),
                "selection_population": "cal", "evaluation_population": "test"})
    pd.DataFrame(rows).to_csv(bank_dir/"final_results.csv", index=False)


def audit(args):
    """Compare nested frozen banks on the same already-frozen calibration Q."""
    root, meta, train, _ = load_instance(args)
    bank_dir, bank = verify_bank(root, meta, args.bank_name)
    _, reference = verify_bank(root, meta, args.reference_bank)
    new_ids = {a["attack_id"]: a["model_sha256"] for a in bank["attacks"]}
    old_ids = {a["attack_id"]: a["model_sha256"] for a in reference["attacks"]}
    if any(new_ids.get(key) != value for key, value in old_ids.items()):
        raise ValueError("the stronger bank must retain every reference full-update model")
    saved = root/"cal_population.pt"
    if file_hash(saved) != saved.with_suffix(".sha256").read_text().strip():
        raise ValueError("frozen calibration population changed")
    population = torch.load(saved, weights_only=True)
    curves, evaluations = bank_curves(bank_dir, bank, population, meta["ratios"],
                                     meta["lambdas"], train["x"].shape[1])
    old = evaluations[evaluations.attack_id.isin(old_ids)]
    old = old.groupby(["radius_ratio", "lambda"], as_index=False).true_hinge.max()
    old = old.rename(columns={"radius_ratio": "r_over_b_ref", "true_hinge": "reference_bank_hinge"})
    comparison = curves.merge(old, on=["r_over_b_ref", "lambda"], validate="one_to_one")
    comparison["risk_increase"] = comparison.pooled_worst_hinge-comparison.reference_bank_hinge
    if (comparison.risk_increase < -1e-7).any():
        raise RuntimeError("a nested bank cannot reduce the risk estimate")
    comparison.to_csv(bank_dir/f"strengthening_vs_{args.reference_bank}.csv", index=False)
    summary = []
    for ratio, rows in comparison.groupby("r_over_b_ref"):
        endpoint = float(rows[np.isclose(rows["lambda"], 1.)].risk_increase.iloc[0])
        others = rows[~np.isclose(rows["lambda"], 1.)].risk_increase
        summary.append({"r_over_b_ref": ratio, "lambda1_risk_increase": endpoint,
            "mean_other_lambda_risk_increase": float(others.mean()),
            "lambda1_excess_increase": endpoint-float(others.mean())})
    pd.DataFrame(summary).to_csv(bank_dir/f"endpoint_audit_vs_{args.reference_bank}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "discover", "freeze", "select", "evaluate", "audit"))
    parser.add_argument("--dataset", choices=legacy.DATASETS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--instance-root", type=Path, default=Path("/home/xu.zhao/robust_ce_results/heloc_diabetes_wro_lambda_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-home", default="/home/xu.zhao/robust_ce_data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--all-grid", action="store_true")
    parser.add_argument("--radius-index", type=int, default=0)
    parser.add_argument("--lambda-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=legacy.SEARCH["steps"])
    parser.add_argument("--generation", type=checked_name, default="solverfix_v3")
    parser.add_argument("--bank-name", type=checked_name, default="bank_v1")
    parser.add_argument("--reference-bank", type=checked_name)
    args = parser.parse_args()
    if args.count <= 0 or args.steps < 0:
        parser.error("count must be positive and steps nonnegative")
    if args.phase == "audit" and args.reference_bank is None:
        parser.error("audit requires --reference-bank")
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
