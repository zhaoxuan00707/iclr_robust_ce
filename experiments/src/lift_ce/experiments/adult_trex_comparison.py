"""Cost-matched Adult comparison of LIFT-CF and paper-faithful T-Rex:I."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from ..counterfactuals.tabular import select_sources
from ..data.tabular import Shift, apply_shift, load_heloc, load_tabular_dataset
from ..models.tabular import train_model, validity


def stability(model, x, k, sigma, seed, noise_batch=100):
    """Eq. (practical relaxation): E[m(x+e)-|m(x)-m(x+e)|]."""
    clean = torch.sigmoid(model(x))
    total = torch.zeros_like(clean)
    generator = torch.Generator().manual_seed(seed)
    done = 0
    while done < k:
        count = min(noise_batch, k - done)
        noise = sigma * torch.randn((count,) + x.shape, generator=generator)
        noisy = x.unsqueeze(0) + noise
        prob = torch.sigmoid(model(noisy.flatten(0, 1))).view(count, len(x))
        total = total + (prob - (clean.unsqueeze(0) - prob).abs()).sum(0)
        done += count
    return total / k


def trex_trajectory(model, initial, k, sigma, lr, steps, seed):
    """Algorithm 1 gradient ascent; retain every iterate for cost matching."""
    model.eval()
    x = initial.clone()
    trajectory = [x.clone()]
    scores = []
    for step in range(steps):
        x = x.detach().requires_grad_(True)
        score = stability(model, x, k, sigma, seed + step)
        score.sum().backward()
        # Algorithm 1 uses direct gradient ascent (no optimizer state).
        x = (x + lr * x.grad).detach()
        trajectory.append(x.clone()); scores.append(score.detach())
        if (step + 1) % 10 == 0:
            print(f"T-Rex step {step+1}/{steps} stability={score.mean():.4f}", flush=True)
    return torch.stack(trajectory), torch.stack(scores)


def mean_l2(samples, sources):
    return float(torch.linalg.vector_norm(samples - sources, ord=2, dim=1).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-home", required=True); p.add_argument("--lift-results", required=True)
    p.add_argument("--dataset", choices=("adult", "heloc"), default="adult")
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--k", type=int, default=1000)
    p.add_argument("--sigma", type=float, default=.1); p.add_argument("--trex-steps", type=int, default=100)
    p.add_argument("--trex-lr", type=float, default=.1)
    p.add_argument("--reuse-trex", action="store_true")
    args = p.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "adult":
        pool_x, pool_y, test_x, test_y, _ = load_tabular_dataset(
            {"source":"openml", "openml_id":1590, "positive_label":">50K"}, args.data_home, .2, args.seed)
    else:
        pool_x, pool_y, test_x, test_y, _ = load_heloc(args.data_home, .2, args.seed)
    ids = np.arange(len(pool_x)); train_ids, _ = train_test_split(
        ids, test_size=.2, random_state=args.seed+1, stratify=pool_y.numpy())
    train_ids = torch.as_tensor(train_ids); train_x, train_y = pool_x[train_ids], pool_y[train_ids]
    base_model, base_acc = train_model(train_x, train_y, test_x, test_y, [64,32], args.epochs, 1e-3, args.seed)
    lift_root = Path(args.lift_results)
    saved_path = lift_root/"counterfactuals.pt"
    if not saved_path.exists():
        saved_path = lift_root/"artifacts.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=True)
    sources = select_sources(base_model, test_x, test_y, 128, args.seed+2)
    base_cf = saved.get("base_cf", saved.get("base_ce"))
    lift_cf = saved.get("lift_cf", saved.get("lift_ce"))

    saved_trex_path = out/"counterfactuals.pt"
    if args.reuse_trex and saved_trex_path.exists():
        previous = torch.load(saved_trex_path, map_location="cpu", weights_only=True)
        trajectory, stability_scores = previous["trex_trajectory"], previous["stability_scores"]
    else:
        trajectory, stability_scores = trex_trajectory(
            base_model, base_cf, args.k, args.sigma, args.trex_lr, args.trex_steps, args.seed+50_000)
    target_cost = mean_l2(lift_cf, sources)
    costs = np.asarray([mean_l2(x, sources) for x in trajectory])
    selected_step = int(np.abs(costs-target_cost).argmin())
    trex_cf = trajectory[selected_step]
    # If T-Rex cannot reach calibrated LIFT's cost within max_steps, lower the
    # LIFT refinement depth on the existing path to T-Rex's achieved L2 cost.
    trex_cost = mean_l2(trex_cf, sources)
    lo, hi = 0., 1.
    for _ in range(40):
        fraction = (lo+hi)/2
        candidate = base_cf + fraction*(lift_cf-base_cf)
        if mean_l2(candidate, sources) < trex_cost: lo = fraction
        else: hi = fraction
    lift_match_fraction = (lo+hi)/2
    lift_l2_matched = base_cf + lift_match_fraction*(lift_cf-base_cf)
    methods = {"base_cf":base_cf, "lift_cf":lift_cf,
               "lift_l2_matched":lift_l2_matched, "trex_i":trex_cf}

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(train_x.numpy())
    rows=[]
    for name,x in methods.items():
        rows.append({"method":name, "current_validity":validity(base_model,x),
                     "mean_l2_cost":mean_l2(x,sources),
                     "mean_lof_score":float(lof.score_samples(x.numpy()).mean()),
                     "lof_inlier_rate":float((lof.predict(x.numpy())==1).mean())})

    sweeps={"mean_shift":[0.,.1,.2,.3,.4,.5,.75,1.,1.25,1.5],
            "scale_shift":[1.,1.1,1.2,1.3,1.4,1.5,1.75,2.,2.5,3.]}
    intensity=[]
    for si,(shift_name,strengths) in enumerate(sweeps.items()):
        for strength in strengths:
            shift=Shift(shift_name,strength)
            sx,sy=apply_shift(train_x,train_y,shift,args.seed+10_000+si)
            tx,ty=apply_shift(test_x,test_y,shift,args.seed+20_000+si)
            future,acc=train_model(sx,sy,tx,ty,[64,32],args.epochs,1e-3,args.seed+30_000+si)
            for name,x in methods.items():
                intensity.append({"shift":shift_name,"intensity":strength,"method":name,
                                  "validity":validity(future,x),"model_accuracy":acc})
            print(f"{shift_name}={strength:g} accuracy={acc:.4f}",flush=True)

    for filename,data in (("summary_metrics.csv",rows),("intensity_metrics.csv",intensity)):
        with (out/filename).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    torch.save({**methods,"trex_trajectory":trajectory,"stability_scores":stability_scores},out/"counterfactuals.pt")
    settings={"dataset":args.dataset,"base_accuracy":base_acc,"k":args.k,"sigma":args.sigma,
              "sigma_squared":args.sigma**2,"trex_lr":args.trex_lr,"trex_steps":args.trex_steps,
              "cost_match_norm":"L2","lift_mean_l2_cost":target_cost,"trex_selected_step":selected_step,
              "trex_mean_l2_cost":trex_cost,"lift_match_fraction":lift_match_fraction,
              "matched_lift_mean_l2_cost":mean_l2(lift_l2_matched,sources),"sweeps":sweeps}
    (out/"settings.json").write_text(json.dumps(settings,indent=2))

    labels={"base_cf":"Base CF","lift_cf":"LIFT-CF (calibrated)","lift_l2_matched":"LIFT-CF (L2-matched)","trex_i":"T-Rex:I"}
    colors={"base_cf":"gray","lift_cf":"tab:blue","lift_l2_matched":"tab:green","trex_i":"tab:red"}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),sharey=True)
    for ax,(shift_name,strengths) in zip(axes,sweeps.items()):
        for method in methods:
            vals=[r["validity"] for r in intensity if r["shift"]==shift_name and r["method"]==method]
            ax.plot(strengths,vals,marker="o",lw=2,label=labels[method],color=colors[method])
        ax.set_title(shift_name.replace("_"," ").title());ax.grid(alpha=.25);ax.set_ylim(-.02,1.02)
        ax.set_xlabel("mean-shift magnitude" if shift_name=="mean_shift" else "scale multiplier")
    axes[0].set_ylabel("future-model counterfactual validity");axes[1].legend()
    fig.suptitle(f"{args.dataset.upper()}: LIFT-CF vs T-Rex:I at matched L2 cost");fig.tight_layout()
    fig.savefig(out/f"{args.dataset}_lift_vs_trex.png",dpi=240,bbox_inches="tight");fig.savefig(out/f"{args.dataset}_lift_vs_trex.pdf",bbox_inches="tight")
    print(json.dumps({"settings":settings,"metrics":rows},indent=2))

if __name__ == "__main__": main()
