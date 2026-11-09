"""Training-only attack strengthening through the complete deployment update."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
from torch.func import functional_call

from . import heloc_diabetes_bank_protocol_v3 as protocol
from .heloc_conditional_adversary_strengthened_v1 import rms, search_loss
from ..counterfactuals.population_path import PopulationPath, population_mean


def differentiable_full_update(model, x, y, delta, q, seed, epochs=8):
    """Replay Update_full's batch order and AdamW steps with outer gradients.

    The zero-variance derivative is defined as zero, retaining AdamW's forward
    denominator rather than changing it by a positive variance floor.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    moments = {name: torch.zeros_like(p) for name, p in params.items()}
    variances = {name: torch.zeros_like(p) for name, p in params.items()}
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed+epoch))
        for start in range(0, len(x), 256):
            index = order[start:start+256].to(x.device)
            logits = functional_call(model, (params, buffers), (x[index]+delta[index],))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[index])
            grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
            step += 1
            updated, next_m, next_v = {}, {}, {}
            for (name, p), grad in zip(params.items(), grads):
                m = .9*moments[name] + .1*grad
                v = .999*variances[name] + .001*grad.square()
                mh, vh = m/(1-.9**step), v/(1-.999**step)
                root = torch.where(vh > 0, torch.sqrt(vh.clamp_min(torch.finfo(vh.dtype).tiny)),
                                   torch.zeros_like(vh))
                updated[name] = p*(1-2e-4*1e-4) - 2e-4*mh/(root+1e-8)
                next_m[name], next_v[name] = m, v
            params, moments, variances = updated, next_m, next_v
    return torch.sigmoid(functional_call(model, (params, buffers), (q,)))


def strengthen(args):
    root, meta, train, current = protocol.load_instance(args)
    population = torch.load(root/"train_population.pt", weights_only=True)
    path = PopulationPath(**population["path"])
    ratio, lam = meta["ratios"][args.radius_index], meta["lambdas"][args.lambda_index]
    output = root/"attacks"/f"{args.generation}_r{ratio:g}_l{lam:.2f}_restart0.pt"
    if output.exists():
        existing = torch.load(output, weights_only=False)
        if existing["instance_id"] != meta["instance_id"] or existing["search"]["steps"] != args.steps:
            raise ValueError("existing generation settings differ")
        return
    # Choose a warm start exclusively by recorded TRAINING source-lambda risk.
    pool = []
    for filename in sorted((root/"attacks").glob("solverfix_v3_*.pt")):
        saved = torch.load(filename, weights_only=False)
        if (saved["instance_id"] != meta["instance_id"] or saved["discovery_split"] != "train"
                or saved["full_update_seed"] != meta["full_update_seed"]):
            raise ValueError("incompatible training warm start")
        if saved["radius_ratio"] == ratio and saved["source_lambda"] == lam:
            for i, candidate in enumerate(saved["candidates"]):
                pool.append((candidate["source_full_metrics"]["true_hinge"],
                             f"{filename.name}:{i}", candidate["delta"]))
    if not pool:
        raise ValueError("no compatible training warm start")
    start_hinge, start_id, initial = max(pool, key=lambda row: (row[0], row[1]))
    radius = ratio*meta["b_ref"]
    model = copy.deepcopy(current).to(args.device).train()
    x, y, q = train["x"].to(args.device), train["y"].to(args.device), path.at(lam).to(args.device)
    delta = initial.to(args.device).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=.075)
    rows, candidates = [], []
    for step in range(args.steps+1):
        pred = differentiable_full_update(model, x, y, delta, q, meta["full_update_seed"])
        objective = search_loss(pred, "softplus", 10., path.weights)
        gradient = torch.autograd.grad(objective, delta)[0]
        finite = bool(torch.isfinite(pred).all() and torch.isfinite(gradient).all())
        if not finite:
            raise RuntimeError(f"nonfinite complete-update derivative at step {step}")
        metrics = {"true_hinge": float(population_mean(torch.relu(.5-pred), path.weights)),
            "invalidity": float(population_mean((pred<.5).float(), path.weights)),
            "budget_fraction": float(rms(delta))/radius}
        if step == 0 and abs(metrics["true_hinge"]-start_hinge) > 1e-4:
            raise RuntimeError("differentiable replay differs from saved official full-update risk")
        rows.append({"iteration": step, "finite_iteration": finite,
                     "search_gradient_norm": float(gradient.norm()), **metrics})
        # Every feasible iterate survives; official Update_full is replayed
        # during freeze and only that model enters calibration ranking.
        candidates.append({"delta": delta.detach().cpu().clone(), "source_full_metrics": metrics})
        print(f"r={ratio:g} lambda={lam:g} step={step} training_hinge={metrics['true_hinge']:.6f}", flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad()
        delta.grad = -gradient
        optimizer.step()
        with torch.no_grad():
            norm = rms(delta)
            if norm > radius:
                delta.mul_(radius/norm)
    protocol.legacy.atomic_torch_save({"instance_id": meta["instance_id"],
        "discovery_split": "train", "source_lambda": lam, "radius_ratio": ratio,
        "restart": 0, "generation": args.generation, "full_update_seed": meta["full_update_seed"],
        "search": {"steps": args.steps, "update": "complete_minibatch_adamw", "warm_start": start_id},
        "candidates": candidates}, output)
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="heloc", choices=protocol.legacy.DATASETS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--radius-index", type=int, default=1)
    parser.add_argument("--lambda-index", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--generation", type=protocol.checked_name, default="full_update_v1")
    parser.add_argument("--device", default="cpu")
    strengthen(parser.parse_args())


if __name__ == "__main__":
    main()
