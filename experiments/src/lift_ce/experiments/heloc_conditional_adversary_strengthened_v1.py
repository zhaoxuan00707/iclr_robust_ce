"""Training-only strengthening of the HELOC (r, lambda)-conditional adversary.

This diagnostic is isolated from production radius/lambda selection.  Search uses
a differentiable full-batch AdamW unroll; every reported adversarial metric uses
the existing official eight-epoch mini-batch AdamW ``fine_tune`` update.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call

from .corrected_future_benchmark_v1 import load_model
from .heloc_coupled_b_lambda_v1 import TAU, score
from .heloc_gradient_sensitive_adversary_v1 import fine_tune
from .heloc_margin_adversary_v4 import rms
from .heloc_prior_balanced_audit import build_shift, predictive_features
from ..data.tabular import load_tabular_three_way
from ..counterfactuals.population_path import population_mean

RADII = (0.5, 1.0, 3.0)
LAMBDA_DIAG = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
LAMBDA_FULL = tuple(np.round(np.linspace(0, 1, 11), 2))
SURROGATES = (("softplus", 1.0), ("softplus", 2.0), ("softplus", 5.0),
              ("softplus", 10.0), ("score", np.nan))
UNROLLS = (1, 3, 5, 10)
LR_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0, 10.0)
STEP_COUNTS = (30, 60, 120)
CURRENT_LR = 0.75
SEED_BASE = 3_460_000
FULL_RESTARTS = 5
RADIAL_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
LOCAL_CENTERS = {0.5: 0.9, 1.0: 0.7, 3.0: 0.8}
LOCAL_REPLICATIONS = 3
LOCAL_STEP = 0.02
LOCAL_NOISE_TOL = 1e-3
LOCAL_PLATEAU_TOL = 5e-3


def tag(x):
    return str(float(x)).replace(".", "p")


def search_loss(pred, kind, kappa, weights=None):
    if kind == "score":
        return -population_mean(pred, weights)
    return population_mean(torch.nn.functional.softplus(kappa * (TAU - pred)), weights) / kappa


def differentiable_adamw_update(model, x, y, delta, q, steps, lr=2e-4,
                                weight_decay=1e-4, beta1=0.9, beta2=0.999,
                                eps=1e-8):
    """Full-batch differentiable AdamW approximation to official Update.

    Moments are reset for every adversarial iteration, just as official
    ``fine_tune`` constructs a fresh optimizer from the frozen base model.
    Mini-batch order/epochs are not reproduced: ``steps`` means full-batch
    AdamW steps and is therefore an explicitly documented approximation.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    moments = {n: torch.zeros_like(p) for n, p in params.items()}
    variances = {n: torch.zeros_like(p) for n, p in params.items()}
    for t in range(1, steps + 1):
        logits = functional_call(model, (params, buffers), (x + delta,))
        task = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        grads = torch.autograd.grad(task, tuple(params.values()), create_graph=True)
        new_params, new_moments, new_variances = {}, {}, {}
        for (name, p), g in zip(params.items(), grads):
            m = beta1 * moments[name] + (1 - beta1) * g
            v = beta2 * variances[name] + (1 - beta2) * g.square()
            mh, vh = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
            # ``vh`` starts at zero for every unroll.  Differentiating
            # ``sqrt(vh)`` at that point can yield NaNs in the *outer*
            # gradient (the forward value is finite, but the second
            # derivative of sqrt is singular at zero).  That made every
            # adversarial search terminate at its 5% initialization.  Keep
            # AdamW semantics while giving the variance a differentiable
            # positive floor.
            new_params[name] = p * (1 - lr * weight_decay) - lr * mh / (torch.sqrt(vh + 1e-12) + eps)
            new_moments[name], new_variances[name] = m, v
        params, moments, variances = new_params, new_moments, new_variances
    pred = torch.sigmoid(functional_call(model, (params, buffers), (q,)))
    return pred


def optimize(search_model, x, y, q, radius, seed, kind, kappa, unroll,
             adversarial_lr, adversarial_steps, weights=None):
    torch.manual_seed(seed)
    delta = torch.randn_like(x)
    delta = (0.05 * radius * delta / rms(delta).clamp_min(1e-12)).requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=adversarial_lr)
    trace = []
    if x.is_cuda:
        torch.cuda.reset_peak_memory_stats(x.device)
    started = time.perf_counter()
    last_feasible = delta.detach().clone()
    for iteration in range(adversarial_steps + 1):
        if not torch.isfinite(delta).all():
            with torch.no_grad():
                delta.copy_(last_feasible)
            break
        pred = differentiable_adamw_update(search_model, x, y, delta, q, unroll)
        objective = search_loss(pred, kind, kappa, weights)
        true_hinge = population_mean(torch.relu(TAU - pred), weights)
        grad = torch.autograd.grad(objective, delta)[0]
        finite = bool(torch.isfinite(pred).all() and torch.isfinite(objective)
                      and torch.isfinite(grad).all())
        norm = float(rms(delta))
        trace.append({
            "iteration": iteration,
            "search_objective_value": float(objective),
            "search_time_true_hinge": float(true_hinge),
            "search_time_invalidity": float(population_mean((pred < TAU).float(), weights)),
            "search_gradient_norm": float(grad.norm()),
            "perturbation_norm": norm,
            "solver_budget_distance": norm,
            "budget_fraction": norm / radius,
            "finite_iteration": finite,
        })
        if not finite or iteration == adversarial_steps:
            break
        last_feasible = delta.detach().clone()
        optimizer.zero_grad()
        delta.grad = -grad
        optimizer.step()
        with torch.no_grad():
            norm_after = rms(delta)
            if norm_after > radius:
                delta.mul_(radius / norm_after)
            if torch.isfinite(delta).all():
                last_feasible = delta.detach().clone()
            else:
                delta.copy_(last_feasible)
                break
    runtime = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(x.device)) if x.is_cuda else np.nan
    return delta.detach().cpu(), pd.DataFrame(trace), runtime, peak


def full_metrics(base_model, train_x, train_y, q, delta, seed, device, weights=None):
    model = fine_tune(base_model, train_x + delta, train_y, seed, device)
    pred = score(model, q)
    return model, {
        "true_hinge": float(population_mean(torch.relu(TAU - pred), weights)),
        "invalidity": float(population_mean((pred < TAU).float(), weights)),
        "validity": float(population_mean((pred >= TAU).float(), weights)),
        "mean_score": float(population_mean(pred, weights)),
    }


def ot_style_distance(train_x, delta, support=128):
    """Same 128-point assignment W2 convention used by future-shift audit."""
    n = min(support, len(train_x))
    a, b = train_x[:n], (train_x + delta)[:n]
    cost = torch.cdist(a, b).square().numpy()
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def radial_full_evaluation(base_model, train_x, train_y, q, delta, radius,
                           seed, device, candidate_sink=None, weights=None):
    before_fraction = float(rms(delta) / radius)
    fractions = sorted(set([before_fraction] + [f for f in RADIAL_FRACTIONS
                                                if f >= before_fraction - 1e-10]))
    unit = delta / rms(delta).clamp_min(1e-12)
    rows, candidates = [], []
    for fraction in fractions:
        candidate = unit * (fraction * radius)
        model, metrics = full_metrics(base_model, train_x, train_y, q, candidate,
                                      seed, device, weights=weights)
        row = {"radial_fraction": fraction, **metrics,
               "solver_budget_distance": float(rms(candidate)),
               "budget_fraction": float(rms(candidate) / radius),
               "ot_style_distance": ot_style_distance(train_x, candidate)}
        rows.append(row)
        candidates.append((row, candidate, model))
        if candidate_sink is not None:
            candidate_sink(row, candidate)
    winner = max(candidates, key=lambda z: (z[0]["true_hinge"], z[0]["invalidity"],
                                             -z[0]["budget_fraction"]))
    return winner, pd.DataFrame(rows)


def run_candidate(base_model, search_model, train_x, train_y, q, radius, seed,
                  kind, kappa, unroll, lr, steps, device):
    delta, trace, runtime, peak = optimize(
        search_model, train_x.to(device), train_y.to(device), q.to(device),
        radius, seed, kind, kappa, unroll, lr, steps)
    winner, radial = radial_full_evaluation(base_model, train_x, train_y, q,
                                             delta, radius, seed + 700_000, device)
    row, selected_delta, selected_model = winner
    before = radial.iloc[int(np.argmin(np.abs(
        radial.radial_fraction.to_numpy() - float(rms(delta) / radius))))]
    summary = {
        "seed": seed, "search_surrogate": kind, "kappa": kappa,
        "unroll_K": unroll, "adversarial_lr": lr, "adversarial_steps": steps,
        "search_objective_final": float(trace.search_objective_value.iloc[-1]),
        "search_gradient_norm_initial": float(trace.search_gradient_norm.iloc[0]),
        "search_gradient_norm_final": float(trace.search_gradient_norm.iloc[-1]),
        "search_gradient_norm_mean": float(trace.search_gradient_norm.mean()),
        "hinge_before_radial": float(before["true_hinge"]),
        "invalidity_before_radial": float(before["invalidity"]),
        "budget_fraction_before_radial": float(rms(delta) / radius),
        "radial_search_improvement": row["true_hinge"] - float(before["true_hinge"]),
        "runtime_seconds": runtime, "peak_memory_bytes": peak, **row,
    }
    return summary, selected_delta, selected_model, trace, radial


def stage_search(base_model, train_x, train_y, tr, bref, out, device):
    """Training-only hyperparameter diagnostic on r=1, lambda=0.9/1."""
    search_model = copy.deepcopy(base_model).to(device).train()
    rows, traces, radial_rows = [], [], []

    # Surrogate comparison at K=3 and current optimizer strength.
    for ci, (kind, kappa) in enumerate(SURROGATES):
        for lam in (0.9, 1.0):
            q = (1 - lam) * tr["base"] + lam * tr["destination"]
            for restart in range(3):
                seed = SEED_BASE + 10_000 * ci + 1000 * int(lam * 10) + restart
                s, _, _, t, radial = run_candidate(base_model, search_model, train_x,
                    train_y, q, bref, seed, kind, kappa, 3, CURRENT_LR, 30, device)
                rows.append({"stage": "surrogate", "lambda": lam,
                             "restart": restart, **s})
                traces.append(t.assign(stage="surrogate", lambda_value=lam,
                                       restart=restart, config=ci))
                radial_rows.append(radial.assign(stage="surrogate", lambda_value=lam,
                                                 restart=restart, config=ci))
    stage = pd.DataFrame(rows)
    surrogate_scores = (stage.groupby(["search_surrogate", "kappa"], dropna=False)
                        .true_hinge.mean().reset_index())
    best_surrogate = surrogate_scores.sort_values("true_hinge", ascending=False).iloc[0]
    best_kind = str(best_surrogate.search_surrogate)
    best_kappa = float(best_surrogate.kappa) if best_kind != "score" else np.nan

    # Unroll comparison with the selected smoother objective.
    for K in UNROLLS:
        for lam in (0.9, 1.0):
            q = (1 - lam) * tr["base"] + lam * tr["destination"]
            for restart in range(2):
                seed = SEED_BASE + 500_000 + 10_000 * K + 1000 * int(lam * 10) + restart
                s, _, _, t, radial = run_candidate(base_model, search_model, train_x,
                    train_y, q, bref, seed, best_kind, best_kappa, K,
                    CURRENT_LR, 30, device)
                rows.append({"stage": "unroll", "lambda": lam,
                             "restart": restart, **s})
                traces.append(t.assign(stage="unroll", lambda_value=lam,
                                       restart=restart, config=K))
                radial_rows.append(radial.assign(stage="unroll", lambda_value=lam,
                                                 restart=restart, config=K))
    stage = pd.DataFrame(rows)
    unroll_scores = (stage[stage.stage == "unroll"].groupby("unroll_K")
                     .true_hinge.mean().reset_index())
    max_score = float(unroll_scores.true_hinge.max())
    # Smallest K within 5% of the best mean final hinge.
    best_K = int(unroll_scores[unroll_scores.true_hinge >= 0.95 * max_score]
                 .unroll_K.min())

    # Optimizer grid at selected surrogate/K.
    for multiplier in LR_MULTIPLIERS:
        for steps in STEP_COUNTS:
            for lam in (0.9, 1.0):
                q = (1 - lam) * tr["base"] + lam * tr["destination"]
                for restart in range(2):
                    seed = (SEED_BASE + 1_000_000 + int(multiplier * 1000)
                            + 10_000 * steps + 100 * int(lam * 10) + restart)
                    s, _, _, t, radial = run_candidate(base_model, search_model,
                        train_x, train_y, q, bref, seed, best_kind, best_kappa,
                        best_K, CURRENT_LR * multiplier, steps, device)
                    rows.append({"stage": "optimizer", "lambda": lam,
                                 "restart": restart, **s})
                    traces.append(t.assign(stage="optimizer", lambda_value=lam,
                                           restart=restart,
                                           config=f"{multiplier}x_{steps}"))
                    radial_rows.append(radial.assign(stage="optimizer", lambda_value=lam,
                                                     restart=restart,
                                                     config=f"{multiplier}x_{steps}"))
    stage = pd.DataFrame(rows)
    opt_scores = (stage[stage.stage == "optimizer"]
                  .groupby(["adversarial_lr", "adversarial_steps"])
                  .true_hinge.mean().reset_index()
                  .sort_values(["true_hinge", "adversarial_steps"],
                               ascending=[False, True]))
    best_opt = opt_scores.iloc[0]
    settings = {"search_surrogate": best_kind,
                "kappa": None if best_kind == "score" else best_kappa,
                "unroll_K": best_K,
                "adversarial_lr": float(best_opt.adversarial_lr),
                "adversarial_steps": int(best_opt.adversarial_steps),
                "selection_basis": "mean full-update true hinge at r/b_ref=1, lambda={0.9,1}; training only"}
    stage.to_csv(out / "stage_comparisons.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out / "stage_optimization_traces.csv", index=False)
    pd.concat(radial_rows, ignore_index=True).to_csv(out / "stage_radial_candidates.csv", index=False)
    (out / "selected_search_settings.json").write_text(json.dumps(settings, indent=2))
    return settings


def full_landscape(base_model, train_x, train_y, tr, bref, settings, old_root,
                   out, device):
    search_model = copy.deepcopy(base_model).to(device).train()
    restart_rows, traces, radial_rows, winners = [], [], [], []
    total = len(RADII) * len(LAMBDA_FULL) * FULL_RESTARTS
    done = 0
    for ri, r in enumerate(RADII):
        radius = r * bref
        for li, lam in enumerate(LAMBDA_FULL):
            q = ((1 - lam) * tr["base"] + lam * tr["destination"]).detach()
            candidates = []
            for restart in range(FULL_RESTARTS):
                seed = SEED_BASE + 2_000_000 + 100_000 * ri + 1000 * li + 17 * restart
                summary, delta, model, trace, radial = run_candidate(
                    base_model, search_model, train_x, train_y, q, radius, seed,
                    settings["search_surrogate"],
                    settings["kappa"] if settings["kappa"] is not None else np.nan,
                    settings["unroll_K"], settings["adversarial_lr"],
                    settings["adversarial_steps"], device)
                summary.update({"dataset": "heloc", "fold": 0, "r": r,
                                "radius": radius, "lambda": lam,
                                "restart": restart, "selected_restart": False})
                restart_rows.append(summary)
                traces.append(trace.assign(r=r, lambda_value=lam, restart=restart,
                                           seed=seed))
                radial_rows.append(radial.assign(r=r, lambda_value=lam,
                                                 restart=restart, seed=seed))
                candidates.append((summary, delta, model))
                done += 1
                print(f"strengthened conditional ADV {done}/{total}", flush=True)
            winner = max(candidates, key=lambda z: (z[0]["true_hinge"],
                                                     z[0]["invalidity"],
                                                     -z[0]["budget_fraction"],
                                                     -z[0]["restart"]))
            restart_rows[-FULL_RESTARTS + winner[0]["restart"]]["selected_restart"] = True
            delta_path = out / "deltas" / f"r{tag(r)}_lambda{tag(lam)}.pt"
            model_path = out / "models" / f"r{tag(r)}_lambda{tag(lam)}.pt"
            torch.save(winner[1], delta_path)
            torch.save(winner[2].state_dict(), model_path)
            row = dict(winner[0])
            row.update({"recourse_cost": float(torch.linalg.vector_norm(
                q - tr["source"], dim=1).mean()), "delta_path": str(delta_path),
                "model_path": str(model_path)})
            winners.append(row)
    restarts = pd.DataFrame(restart_rows)
    curve = pd.DataFrame(winners).sort_values(["r", "lambda"])
    old = pd.read_csv(old_root / "conditional_phi_curves.csv")
    old = old[["r", "lambda", "phi_hinge", "budget_usage"]].rename(
        columns={"phi_hinge": "old_phi_hinge", "budget_usage": "old_budget_fraction"})
    curve = curve.merge(old, on=["r", "lambda"], how="left", validate="one_to_one")
    curve["delta_hinge_new_minus_old"] = curve.true_hinge - curve.old_phi_hinge
    curve["delta_budget_fraction_new_minus_old"] = (curve.budget_fraction
                                                      - curve.old_budget_fraction)
    restarts.to_csv(out / "strengthened_restarts.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(out / "strengthened_optimization_traces.csv", index=False)
    pd.concat(radial_rows, ignore_index=True).to_csv(out / "strengthened_radial_candidates.csv", index=False)
    curve.to_csv(out / "strengthened_phi_curves.csv", index=False)
    return curve


def matched_catastrophe(base_model, train_x, train_y, tr, future_specs, bref,
                        out, device):
    spec = pd.read_csv(future_specs)
    row = spec[spec.shift_id == "target_side_shift|s2|k6"].iloc[0]
    features, _ = predictive_features(train_x, train_y)
    shifted_x, shifted_y, kind = build_shift(
        row.shift_family, train_x, train_y, float(row.generation_severity), 6,
        int(row.seed), features)
    assert torch.equal(shifted_y, train_y) and len(shifted_x) == len(train_x)
    delta = shifted_x - train_x
    matched = fine_tune(base_model, shifted_x, shifted_y, int(row.training_seed), device)
    mismatched = load_model(Path(row.model_path), train_x.shape[1])
    solver_distance = float(rms(delta))
    ot_distance = ot_style_distance(train_x, delta)
    rows = []
    for r in RADII:
        for lam in LAMBDA_FULL:
            q = (1 - lam) * tr["base"] + lam * tr["destination"]
            pm, po = score(matched, q), score(mismatched, q)
            rows.append({"shift_id": row.shift_id, "shift_kind": kind, "r": r,
                         "lambda": lam, "solver_budget_distance": solver_distance,
                         "solver_budget_fraction": solver_distance / (r * bref),
                         "ot_style_distance": ot_distance,
                         "ot_budget_fraction": ot_distance / (r * bref),
                         "inside_solver_proxy_ball": solver_distance <= r * bref + 1e-10,
                         "inside_ot_style_ball": ot_distance <= r * bref + 1e-10,
                         "matched_update_hinge": float(torch.relu(TAU - pm).mean()),
                         "matched_update_invalidity": float((pm < TAU).float().mean()),
                         "old_mismatched_protocol_hinge": float(torch.relu(TAU - po).mean()),
                         "old_mismatched_protocol_invalidity": float((po < TAU).float().mean())})
    result = pd.DataFrame(rows)
    result.to_csv(out / "matched_catastrophic_witness.csv", index=False)
    return result


def local_replication(base_model, train_x, train_y, tr, bref, settings, out,
                      device):
    """Independent-seed local lambda refinement; production selectors untouched."""
    local_out = out / "local_replication"
    local_out.mkdir(parents=True, exist_ok=True)
    search_model = copy.deepcopy(base_model).to(device).train()
    rows, restart_rows, trace_rows, radial_rows = [], [], [], []
    total = sum(len(set(np.round(np.arange(max(0, c - .1), min(1, c + .1)
        + LOCAL_STEP / 2, LOCAL_STEP), 2).tolist() + [1.0]))
        for c in LOCAL_CENTERS.values()) * LOCAL_REPLICATIONS * FULL_RESTARTS
    done = 0
    for replication in range(LOCAL_REPLICATIONS):
        for ri, r in enumerate(RADII):
            center = LOCAL_CENTERS[r]
            lambdas = sorted(set(np.round(np.arange(max(0, center - .1),
                min(1, center + .1) + LOCAL_STEP / 2, LOCAL_STEP), 2).tolist()
                + [1.0]))
            radius = r * bref
            for li, lam in enumerate(lambdas):
                q = ((1 - lam) * tr["base"] + lam * tr["destination"]).detach()
                candidates = []
                for restart in range(FULL_RESTARTS):
                    seed = (SEED_BASE + 7_000_000 + 1_000_000 * replication
                            + 100_000 * ri + 1000 * li + 17 * restart)
                    summary, delta, model, trace, radial = run_candidate(
                        base_model, search_model, train_x, train_y, q, radius, seed,
                        settings["search_surrogate"],
                        settings["kappa"] if settings["kappa"] is not None else np.nan,
                        settings["unroll_K"], settings["adversarial_lr"],
                        settings["adversarial_steps"], device)
                    summary.update({"dataset": "heloc", "fold": 0,
                        "replication": replication, "r": r, "radius": radius,
                        "lambda": lam, "restart": restart,
                        "selected_restart": False,
                        "search_all_iterations_finite": bool(trace.finite_iteration.all())})
                    restart_rows.append(summary)
                    candidates.append((summary, delta, model))
                    trace_rows.append(trace.assign(replication=replication, r=r,
                        lambda_value=lam, restart=restart, seed=seed))
                    radial_rows.append(radial.assign(replication=replication, r=r,
                        lambda_value=lam, restart=restart, seed=seed))
                    done += 1
                    print(f"local replication {done}/{total}", flush=True)
                winner = max(candidates, key=lambda z: (z[0]["true_hinge"],
                    z[0]["invalidity"], -z[0]["budget_fraction"],
                    -z[0]["restart"]))
                restart_rows[-FULL_RESTARTS + winner[0]["restart"]]["selected_restart"] = True
                row = dict(winner[0])
                row["recourse_cost"] = float(torch.linalg.vector_norm(
                    q - tr["source"], dim=1).mean())
                rows.append(row)
    curves = pd.DataFrame(rows).sort_values(["r", "replication", "lambda"])
    restarts = pd.DataFrame(restart_rows)
    curves.to_csv(local_out / "local_phi_curves.csv", index=False)
    restarts.to_csv(local_out / "local_restarts.csv", index=False)
    pd.concat(trace_rows, ignore_index=True).to_csv(
        local_out / "local_optimization_traces.csv", index=False)
    pd.concat(radial_rows, ignore_index=True).to_csv(
        local_out / "local_radial_candidates.csv", index=False)
    return curves


def analyze_local(curves, out):
    local_out = out / "local_replication"
    selections, neighbor_rows = [], []
    for (r, replication), group in curves.groupby(["r", "replication"]):
        group = group.sort_values("lambda").reset_index(drop=True)
        best_index = int(group.true_hinge.argmin())
        best = group.iloc[best_index]
        interior = group[group["lambda"] < 1].sort_values("true_hinge").iloc[0]
        endpoint = group[np.isclose(group["lambda"], 1)].iloc[0]
        local_only = group[group["lambda"] < 1].sort_values("lambda").reset_index(drop=True)
        local_index = int(local_only.index[local_only["lambda"].eq(best["lambda"])][0]) if best["lambda"] < 1 else -1
        left = local_only.iloc[local_index - 1] if local_index > 0 else None
        right = local_only.iloc[local_index + 1] if 0 <= local_index < len(local_only) - 1 else None
        left_gap = float(left.true_hinge - best.true_hinge) if left is not None else np.nan
        right_gap = float(right.true_hinge - best.true_hinge) if right is not None else np.nan
        meaningful_neighbors = bool(left is not None and right is not None
            and left_gap > LOCAL_NOISE_TOL and right_gap > LOCAL_NOISE_TOL)
        plateau = group[group.true_hinge <= float(best.true_hinge) + LOCAL_PLATEAU_TOL]
        plateau_width = float(plateau["lambda"].max() - plateau["lambda"].min())
        if best["lambda"] >= 0.9 and float(endpoint.true_hinge - best.true_hinge) <= LOCAL_NOISE_TOL:
            curve_classification = "C. monotone / endpoint-seeking"
        elif plateau_width >= .04:
            curve_classification = "B. broad interior plateau"
        elif best["lambda"] < 1 and meaningful_neighbors:
            curve_classification = "A. stable interior minimum within this replication"
        else:
            curve_classification = "D. noisy / unstable"
        selections.append({"r": r, "replication": replication,
            "lambda_rob": float(best["lambda"]), "minimum_hinge": float(best.true_hinge),
            "minimum_invalidity": float(best.invalidity),
            "minimum_validity": float(best.validity),
            "minimum_cost": float(best.recourse_cost),
            "minimum_budget_fraction": float(best.budget_fraction),
            "selected_restart": int(best.restart),
            "left_neighbor_lambda": float(left["lambda"]) if left is not None else np.nan,
            "left_neighbor_hinge": float(left.true_hinge) if left is not None else np.nan,
            "right_neighbor_lambda": float(right["lambda"]) if right is not None else np.nan,
            "right_neighbor_hinge": float(right.true_hinge) if right is not None else np.nan,
            "left_hinge_gap": left_gap, "right_hinge_gap": right_gap,
            "meaningfully_worse_both_neighbors": meaningful_neighbors,
            "plateau_lambda_min": float(plateau["lambda"].min()),
            "plateau_lambda_max": float(plateau["lambda"].max()),
            "curve_classification": curve_classification,
            "endpoint_hinge": float(endpoint.true_hinge),
            "endpoint_cost": float(endpoint.recourse_cost),
            "best_interior_lambda": float(interior["lambda"]),
            "best_interior_hinge": float(interior.true_hinge),
            "best_interior_cost": float(interior.recourse_cost),
            "delta_phi_endpoint_minus_interior": float(endpoint.true_hinge - interior.true_hinge),
            "delta_cost_endpoint_minus_interior": float(endpoint.recourse_cost - interior.recourse_cost)})
        neighbor_rows.extend([{"r": r, "replication": replication,
            "position": position, "lambda": float(point["lambda"]),
            "true_hinge": float(point.true_hinge)} for position, point in
            (("left", left), ("minimum", best), ("right", right)) if point is not None])
    selections = pd.DataFrame(selections)
    summaries, classifications = [], []
    for r, group in selections.groupby("r"):
        all_interior = bool((group.lambda_rob < 1).all())
        recurring = float(group.lambda_rob.max() - group.lambda_rob.min()) <= 0.06
        sharp_fraction = float(group.meaningfully_worse_both_neighbors.mean())
        plateau_width = float((group.plateau_lambda_max - group.plateau_lambda_min).mean())
        near_endpoint_fraction = float((group.lambda_rob >= .9).mean())
        if all_interior and recurring and sharp_fraction >= 2 / 3:
            classification = "A. stable interior minimum"
        elif all_interior and recurring and plateau_width >= .04:
            classification = "B. broad interior plateau"
        elif near_endpoint_fraction >= 2 / 3:
            classification = "C. monotone / endpoint-seeking"
        else:
            classification = "D. noisy / unstable"
        summaries.append({"r": r, "mean_lambda_rob": float(group.lambda_rob.mean()),
            "median_lambda_rob": float(group.lambda_rob.median()),
            "std_lambda_rob": float(group.lambda_rob.std(ddof=0)),
            "min_lambda_rob": float(group.lambda_rob.min()),
            "max_lambda_rob": float(group.lambda_rob.max()),
            "fraction_lambda_equal_1": float(np.isclose(group.lambda_rob, 1).mean()),
            "fraction_lambda_ge_0p9": near_endpoint_fraction,
            "mean_delta_phi_endpoint_minus_interior": float(group.delta_phi_endpoint_minus_interior.mean()),
            "mean_delta_cost_endpoint_minus_interior": float(group.delta_cost_endpoint_minus_interior.mean()),
            "classification": classification})
        classifications.append(classification)
    summaries = pd.DataFrame(summaries)
    selections.to_csv(local_out / "local_robustness_minimizers.csv", index=False)
    summaries.to_csv(local_out / "local_stability_summary.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(local_out / "local_neighbor_checks.csv", index=False)
    for r, group in curves.groupby("r"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for replication, q in group.groupby("replication"):
            q = q.sort_values("lambda")
            axes[0].plot(q["lambda"], q.true_hinge, marker="o",
                         label=f"seed replication {replication}")
            axes[1].plot(q["lambda"], q.recourse_cost, marker="o",
                         label=f"seed replication {replication}")
        axes[0].set(xlabel="lambda", ylabel="empirical lower-bound hinge")
        axes[1].set(xlabel="lambda", ylabel="recourse cost")
        for axis in axes:
            axis.grid(alpha=.25); axis.legend(fontsize=8)
        fig.suptitle(f"HELOC local replication r/b_ref={r:g}")
        fig.tight_layout(); fig.savefig(local_out / f"r{tag(r)}_local_replications.png", dpi=180)
        plt.close(fig)
    return selections, summaries


def report(curve, witness, out):
    # Repair runs produced before same-seed before/after reporting was enforced.
    radial_path, restart_path = (out / "strengthened_radial_candidates.csv",
                                 out / "strengthened_restarts.csv")
    if radial_path.exists() and restart_path.exists():
        radial = pd.read_csv(radial_path)
        optimization = pd.read_csv(out / "strengthened_optimization_traces.csv")
        finite = (optimization.groupby(["r", "lambda_value", "restart"])
                  .finite_iteration.all().reset_index()
                  .rename(columns={"lambda_value": "lambda",
                                   "finite_iteration": "search_all_iterations_finite"}))
        before = (radial.sort_values("radial_fraction")
                  .groupby(["r", "lambda_value", "restart"], as_index=False).first()
                  [["r", "lambda_value", "restart", "true_hinge", "invalidity",
                    "budget_fraction"]]
                  .rename(columns={"lambda_value": "lambda",
                    "true_hinge": "same_seed_hinge_before_radial",
                    "invalidity": "same_seed_invalidity_before_radial",
                    "budget_fraction": "same_seed_budget_before_radial"}))
        restarts = pd.read_csv(restart_path).drop(columns=[
            "same_seed_hinge_before_radial", "same_seed_invalidity_before_radial",
            "same_seed_budget_before_radial"], errors="ignore")
        restarts = restarts.merge(before, on=["r", "lambda", "restart"],
                                  how="left", validate="one_to_one")
        restarts = restarts.drop(columns=["search_all_iterations_finite"], errors="ignore")
        restarts = restarts.merge(finite, on=["r", "lambda", "restart"],
                                  how="left", validate="one_to_one")
        restarts["hinge_before_radial"] = restarts.same_seed_hinge_before_radial
        restarts["invalidity_before_radial"] = restarts.same_seed_invalidity_before_radial
        restarts["budget_fraction_before_radial"] = restarts.same_seed_budget_before_radial
        restarts["radial_search_improvement"] = (restarts.true_hinge
                                                  - restarts.hinge_before_radial)
        restarts.to_csv(restart_path, index=False)
        selected = restarts[restarts.selected_restart][[
            "r", "lambda", "restart", "hinge_before_radial",
            "invalidity_before_radial", "budget_fraction_before_radial",
            "radial_search_improvement", "search_all_iterations_finite"]]
        curve = curve.drop(columns=["hinge_before_radial", "invalidity_before_radial",
            "budget_fraction_before_radial", "radial_search_improvement"], errors="ignore")
        curve = curve.merge(selected, on=["r", "lambda", "restart"],
                            how="left", validate="one_to_one")
        curve.to_csv(out / "strengthened_phi_curves.csv", index=False)
    witness_key = witness[["r", "lambda", "matched_update_hinge",
                           "inside_solver_proxy_ball", "inside_ot_style_ball"]]
    merged = curve.merge(witness_key, on=["r", "lambda"], validate="one_to_one")
    merged["matched_witness_minus_new"] = np.where(
        merged.inside_solver_proxy_ball, merged.matched_update_hinge - merged.true_hinge,
        np.nan)
    merged.to_csv(out / "strengthened_inner_gap.csv", index=False)
    classifications = []
    tol = 1e-4
    for r, group in curve.groupby("r"):
        group = group.sort_values("lambda")
        values = group.true_hinge.to_numpy()
        changes = np.diff(values)
        best = float(values.min())
        minimizers = group[group.true_hinge <= best + tol]
        exact = group.iloc[int(np.argmin(values))]
        if np.all(changes <= tol) and exact["lambda"] >= 0.9:
            label = "monotone toward lambda=1" if exact["lambda"] == 1 else "approximately monotone / high-lambda plateau"
        elif exact["lambda"] < 1 and 0 < exact.name and len(minimizers) == 1:
            label = "apparent interior minimum; replication/local refinement required"
        else:
            label = "noisy / inconclusive"
        classifications.append({"r": r, "minimum_hinge": best,
                                "exact_minimizer_lambda": exact["lambda"],
                                "tied_lambdas_tolerance_1e-4": ",".join(map(str, minimizers["lambda"])),
                                "classification": label})
    pd.DataFrame(classifications).to_csv(out / "endpoint_diagnosis.csv", index=False)
    for r, group in curve.groupby("r"):
        group = group.sort_values("lambda")
        for name, x, y, xlabel, ylabel in (
            ("phi_vs_lambda", "lambda", "true_hinge", "lambda", "empirical lower-bound hinge"),
            ("budget_vs_lambda", "lambda", "budget_fraction", "lambda", "solver budget fraction"),
            ("cost_vs_lambda", "lambda", "recourse_cost", "lambda", "recourse cost"),
            ("phi_vs_cost", "recourse_cost", "true_hinge", "recourse cost", "empirical lower-bound hinge")):
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            ax.plot(group[x], group[y], marker="o", label="strengthened conditional")
            if name in {"phi_vs_lambda", "phi_vs_cost"}:
                ax.plot(group[x], group.old_phi_hinge, marker="s", ls="--", label="old conditional")
            ax.set(xlabel=xlabel, ylabel=ylabel, title=f"HELOC r/b_ref={r:g}")
            ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
            fig.savefig(out / f"r{tag(r)}_{name}.png", dpi=180)
            plt.close(fig)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("stage", "full", "report", "local", "local-report", "all"), default="all")
    parser.add_argument("--data-home", type=Path, default=Path("/mnt/big_disk/xu.zhao/robust_ce_data/openml"))
    parser.add_argument("--current-root", type=Path, default=Path("/mnt/big_disk/xu.zhao/robust_ce_results/heloc_benchmark_milestone_seed0"))
    parser.add_argument("--coupled-root", type=Path, default=Path("/home/xu.zhao/robust_ce_results/heloc_coupled_b_lambda_v1"))
    parser.add_argument("--training-path", type=Path, default=Path("/home/xu.zhao/robust_ce_results/heloc_margin_adversary_v3/training_cf_path.pt"))
    parser.add_argument("--old-root", type=Path, default=Path("/home/xu.zhao/robust_ce_results/heloc_conditional_adversary_landscape_v1"))
    parser.add_argument("--future-specs", type=Path, default=Path("/home/xu.zhao/robust_ce_results/heloc_expanded_inball_common_eval_v1/expanded_per_shift_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("/home/xu.zhao/robust_ce_results/heloc_conditional_adversary_strengthened_v1"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "deltas").mkdir(exist_ok=True)
    (args.output / "models").mkdir(exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = load_tabular_three_way({"source": "openml", "openml_name": "heloc",
        "openml_version": 1, "missing_values": [-7, -8, -9]}, str(args.data_home), 0.6, 0.2, 0)
    train_x, train_y = data["train_x"], data["train_y"]
    base_model = load_model(args.current_root / "models/current.pt", train_x.shape[1])
    tr = torch.load(args.training_path, map_location="cpu", weights_only=True)
    bref = float(json.loads((args.coupled_root / "metadata.json").read_text())["b_ref"])
    started = time.perf_counter()
    if args.phase in {"stage", "all"}:
        settings = stage_search(base_model, train_x, train_y, tr, bref, args.output, device)
    else:
        settings = json.loads((args.output / "selected_search_settings.json").read_text())
    if args.phase in {"local", "local-report"}:
        if args.phase == "local":
            local_curves = local_replication(base_model, train_x, train_y, tr,
                bref, settings, args.output, device)
        else:
            local_curves = pd.read_csv(args.output / "local_replication" / "local_phi_curves.csv")
        selections, summaries = analyze_local(local_curves, args.output)
        print("\nLOCAL MINIMIZERS\n", selections.to_string(index=False))
        print("\nSTABILITY\n", summaries.to_string(index=False))
        return
    if args.phase in {"full", "all"}:
        curve = full_landscape(base_model, train_x, train_y, tr, bref, settings,
                               args.old_root, args.output, device)
        witness = matched_catastrophe(base_model, train_x, train_y, tr,
                                      args.future_specs, bref, args.output, device)
    elif args.phase == "report":
        curve = pd.read_csv(args.output / "strengthened_phi_curves.csv")
        witness = pd.read_csv(args.output / "matched_catastrophic_witness.csv")
    else:
        curve = witness = None
    if args.phase in {"full", "report", "all"}:
        gap = report(curve, witness, args.output)
        metadata = {"diagnostic_only": True, "production_selector_modified": False,
            "future_used_for_search_or_setting_selection": False,
            "parameterization": "fixed-weight, fixed-label sample-wise Delta",
            "search_update": "differentiable full-batch AdamW unroll from frozen f_base",
            "full_evaluation_update": "official 8-epoch mini-batch AdamW from frozen f_base",
            "solver_feasible_set": "identity-coupling RMS proxy subset of/equal-feasible for W2; not full U_r",
            "radii_over_b_ref": RADII, "lambda_grid": LAMBDA_FULL,
            "restarts": FULL_RESTARTS, "settings": settings, "b_ref": bref,
            "runtime_seconds_this_invocation": time.perf_counter() - started}
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print("\nCURVES\n", curve[["r", "lambda", "true_hinge", "invalidity",
            "recourse_cost", "budget_fraction", "ot_style_distance",
            "old_phi_hinge", "delta_hinge_new_minus_old"]].to_string(index=False))
        print("\nGAP\n", gap.groupby("r").matched_witness_minus_new.agg(["min", "mean", "max"]).to_string())
    else:
        print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
