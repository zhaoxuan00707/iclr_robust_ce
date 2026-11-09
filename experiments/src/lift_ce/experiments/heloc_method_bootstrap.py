"""Method-aligned HELOC bootstrap calibration experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

from ..counterfactuals.tabular import generate_wachter_ce, reference_targets, select_sources
from ..data.tabular import Shift, apply_shift, load_heloc, load_tabular_dataset
from ..models.tabular import train_model, validity


def exact_w2(x: torch.Tensor, y: torch.Tensor) -> float:
    cost = torch.cdist(x, y).pow(2).numpy()
    rows, columns = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, columns].mean()))


def bootstrap_indices(n: int, count: int, seed: int) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    return [torch.as_tensor(rng.integers(0, n, n)) for _ in range(count)]


def fixed_path(base_ce, targets, lambdas):
    cost = torch.cdist(base_ce, targets).pow(2).numpy()
    rows, columns = linear_sum_assignment(cost)
    match = torch.empty(len(base_ce), dtype=torch.long)
    match[torch.as_tensor(rows)] = torch.as_tensor(columns)
    destination = targets[match]
    return [(1 - lam) * base_ce + lam * destination for lam in lambdas], match


def calibration_stress(values, budget, seed):
    """Generate a diverse sample-level stress satisfying Eq. calibration_budget."""
    generator = torch.Generator().manual_seed(seed)
    delta = torch.randn(values.shape, generator=generator)
    # Use varying radii inside the common ball, while always respecting it.
    radius = budget * (0.5 + 0.5 * float(torch.rand((), generator=generator)))
    rms = delta.square().sum(1).mean().sqrt().clamp_min(1e-12)
    delta = delta * (radius / rms)
    return values + delta, float(delta.square().sum(1).mean().sqrt())


def train_proxy(member, train_x, train_y, cal_x, cal_y, test_x, test_y,
                candidates, budget, epochs, seed):
    stressed_cal_x, realized_budget = calibration_stress(
        cal_x, budget, seed + 10_000 + member
    )
    model, accuracy = train_model(
        torch.cat((train_x, stressed_cal_x)), torch.cat((train_y, cal_y)),
        test_x, test_y, [64, 32], epochs, 1e-3, seed + member,
    )
    losses = []
    with torch.no_grad():
        for candidate in candidates:
            score = torch.sigmoid(model(candidate))
            losses.append(float(torch.relu(torch.tensor(.5) - score).mean()))
    return member, accuracy, realized_budget, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-home", required=True); p.add_argument("--output", required=True)
    p.add_argument("--dataset", choices=("heloc", "adult"), default="heloc")
    p.add_argument("--B", type=int, default=50); p.add_argument("--M", type=int, default=50)
    p.add_argument("--workers", type=int, default=7); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--ce-count", type=int, default=128)
    p.add_argument("--cal-fraction", type=float, default=.2); p.add_argument("--bootstrap-support", type=int, default=128)
    args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "adult":
        pool_x, pool_y, test_x, test_y, features = load_tabular_dataset(
            {"source": "openml", "openml_id": 1590, "positive_label": ">50K"},
            args.data_home, .2, args.seed,
        )
    else:
        pool_x, pool_y, test_x, test_y, features = load_heloc(args.data_home, .2, args.seed)
    ids = np.arange(len(pool_x))
    train_ids, cal_ids = train_test_split(
        ids, test_size=args.cal_fraction, random_state=args.seed + 1,
        stratify=pool_y.numpy(),
    )
    train_ids, cal_ids = torch.as_tensor(train_ids), torch.as_tensor(cal_ids)
    train_x, train_y = pool_x[train_ids], pool_y[train_ids]
    cal_x, cal_y = pool_x[cal_ids], pool_y[cal_ids]
    base, base_accuracy = train_model(train_x, train_y, test_x, test_y, [64, 32], args.epochs, 1e-3, args.seed)
    sources = select_sources(base, test_x, test_y, args.ce_count, args.seed + 2)
    targets = reference_targets(train_x, train_y, args.ce_count, args.seed + 3, base)
    base_ce = generate_wachter_ce(base, sources, .5, 300, .03, .05, 20.0)

    support_gen = torch.Generator().manual_seed(args.seed + 4)
    support = torch.randperm(len(cal_x), generator=support_gen)[:args.bootstrap_support]
    budget_indices = bootstrap_indices(len(support), args.B, args.seed + 5)
    bootstrap_w2 = [exact_w2(cal_x[support][idx], cal_x[support]) for idx in budget_indices]
    b_cal = float(np.quantile(bootstrap_w2, .95))

    lambdas = [i / 20 for i in range(21)]
    candidates, match = fixed_path(base_ce, targets, lambdas)
    proxy_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(
            train_proxy, member, train_x, train_y, cal_x, cal_y,
            test_x, test_y, candidates, b_cal, args.epochs, args.seed + 1000,
        ) for member in range(args.M)]
        for job in concurrent.futures.as_completed(jobs):
            proxy_rows.append(job.result())
    proxy_rows.sort()
    proxy_losses = np.asarray([row[3] for row in proxy_rows])
    risks = proxy_losses.max(0)
    # Select on the robustness--refinement frontier. Lambda is already the
    # normalized W2 refinement depth, so no second raw-distance penalty enters.
    total_gain = max(float(risks[0] - risks[-1]), 1e-12)
    normalized_gain = (risks[0] - risks) / total_gain
    surplus = normalized_gain - np.asarray(lambdas)
    chosen_index = int(np.argmax(surplus)); chosen = lambdas[chosen_index]
    lift_ce = candidates[chosen_index]

    future_specs = [Shift("mean_shift", .15), Shift("scale_shift", 1.15), Shift("adverse_tail", .25)]
    future_rows = []
    for index, shift in enumerate(future_specs):
        shifted_train_x, shifted_train_y = apply_shift(train_x, train_y, shift, args.seed + 200 + index)
        shifted_test_x, shifted_test_y = apply_shift(test_x, test_y, shift, args.seed + 300 + index)
        future, acc = train_model(shifted_train_x, shifted_train_y, shifted_test_x, shifted_test_y,
                                  [64, 32], args.epochs, 1e-3, args.seed + 2000 + index)
        future_rows.append({"shift": shift.name, "strength": shift.strength,
                            "model_accuracy": acc, "base_cf_validity": validity(future, base_ce),
                            "lift_cf_validity": validity(future, lift_ce)})

    summary = {
        "dataset": args.dataset, "B": args.B, "M": args.M,
        "proxy_type": "bounded_sample_level_calibration_stress",
        "b_cal": b_cal, "bootstrap_quantile": .95, "lambda_star": chosen,
        "lambda_selection": "max_normalized_robustness_gain_minus_lambda",
        "raw_risk_minimizer": lambdas[int(np.argmin(risks))],
        "stress_type": "independent_gaussian_sample_perturbations",
        "realized_stress_budgets": [row[2] for row in proxy_rows],
        "base_model_accuracy": base_accuracy,
        "base_current_validity": validity(base, base_ce), "lift_current_validity": validity(base, lift_ce),
        "base_cost": float(torch.linalg.vector_norm(base_ce - sources, dim=1).mean()),
        "lift_cost": float(torch.linalg.vector_norm(lift_ce - sources, dim=1).mean()),
        "anchor_w2": exact_w2(base_ce, targets[match]), "future": future_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "calibration_curve.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["lambda", "worst_proxy_margin_invalidity"]); w.writerows(zip(lambdas, risks))
    with (out / "bootstrap_proxy_losses.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["proxy", "accuracy", "realized_budget", *[f"lambda_{x:g}" for x in lambdas]])
        w.writerows((member, acc, budget, *losses) for member, acc, budget, losses in proxy_rows)
    with (out / "future_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=future_rows[0]); w.writeheader(); w.writerows(future_rows)
    torch.save({"sources": sources, "base_ce": base_ce, "lift_ce": lift_ce,
                "targets": targets, "match": match, "feature_names": features,
                "bootstrap_w2": bootstrap_w2}, out / "artifacts.pt")

    fig, ax = plt.subplots(figsize=(5.2, 3.5)); ax.plot(lambdas, risks, marker="o")
    ax.axvline(chosen, color="tab:red", linestyle="--", label=f"λ*={chosen:g}")
    ax.set(xlabel="λ", ylabel="worst proxy margin invalidity", title=f"{args.dataset.upper()} bootstrap calibration")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out / "calibration_risk.png", dpi=220); plt.close(fig)
    delta = (lift_ce - base_ce).abs().mean(0); top = torch.topk(delta, min(12, len(delta)))
    fig, ax = plt.subplots(figsize=(7, 4)); ax.barh([features[i] for i in top.indices.flip(0)], top.values.flip(0))
    ax.set(xlabel="mean |LIFT-CF − base CF|", title=f"Largest {args.dataset.upper()} feature refinements")
    fig.tight_layout(); fig.savefig(out / "feature_refinement.png", dpi=220); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
