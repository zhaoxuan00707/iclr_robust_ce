"""Run official RobustX PROPLACE on the frozen main-table artifacts.

The only RobustX modification permitted by this driver is the separately recorded
public-gurobipy import compatibility patch.  No fallback generator is provided.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gurobipy as gp
from sklearn.neighbors import LocalOutlierFactor

from robustx.datasets.DatasetLoader import DatasetLoader
from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.lib.tasks.ClassificationTask import ClassificationTask

from .finish_ellice_geo_german import (
    evaluate,
    future_models,
    load_common,
    schema_from_training,
    validate_cf,
)
from .heloc_benchmark_milestone import valid


class FrozenFrameDataset(DatasetLoader):
    @property
    def X(self):
        return self.data.drop(columns=["target"])

    @property
    def y(self):
        return self.data[["target"]]


def robustx_model_from_frozen(current: torch.nn.Module, input_dim: int) -> SimpleNNModel:
    """Copy the frozen [64,32] logit MLP into RobustX's sigmoid wrapper exactly."""
    wrapped = SimpleNNModel(input_dim, [64, 32], 1, seed=0)
    source_layers = [layer for layer in current.features if isinstance(layer, torch.nn.Linear)]
    source_layers.append(current.classifier)
    target_layers = [layer for layer in wrapped.model if isinstance(layer, torch.nn.Linear)]
    if len(source_layers) != len(target_layers):
        raise RuntimeError("Frozen classifier and RobustX MLP architectures differ")
    with torch.no_grad():
        for source, target in zip(source_layers, target_layers):
            target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
    wrapped.model.eval()
    return wrapped


def run_dataset(ds, home, roots, german_root, future_root, output):
    d, source, ids, _, current, _, _ = load_common(ds, home, roots, german_root)
    train = d["train_x"]
    names = list(d["feature_names"])
    rx_model = robustx_model_from_frozen(current, train.shape[1])
    with torch.no_grad():
        reference = torch.sigmoid(current(train[:64])).numpy()
    copied = rx_model.predict(pd.DataFrame(train[:64].numpy(), columns=names)).to_numpy().ravel()
    max_score_error = float(np.max(np.abs(reference - copied)))
    if max_score_error > 1e-6:
        raise RuntimeError(f"RobustX model copy changed scores: max error {max_score_error}")

    data = FrozenFrameDataset(seed=0)
    frame = pd.DataFrame(train.numpy(), columns=names)
    frame["target"] = d["train_y"].numpy()
    data.data = frame
    generator = PROPLACE(ClassificationTask(rx_model, data))

    cache = output / f"proplace_{ds}_counterfactuals.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        rows, failed = list(saved["counterfactuals"]), list(saved["failed"].bool())
    else:
        rows, failed = [], []

    for i in range(len(rows), len(source)):
        factual = pd.Series(source[i].numpy(), index=names)
        try:
            result = generator.generate_for_instance(factual)
            if result.empty or result.shape[1] != len(names):
                raise RuntimeError("PROPLACE returned an empty or malformed counterfactual")
            value = torch.tensor(result.iloc[0].to_numpy(float), dtype=torch.float32)
            bad = not bool(torch.isfinite(value).all())
        except gp.GurobiError:
            # Solver/API/license failures are experiment-wide blockers.  They must
            # not be converted into per-instance generation failures.
            raise
        except Exception as exc:
            print(f"PROPLACE {ds} source {i} failed: {type(exc).__name__}: {exc}", flush=True)
            value, bad = source[i].clone(), True
        rows.append(value)
        failed.append(bad)
        torch.save(
            {"counterfactuals": torch.stack(rows), "failed": torch.tensor(failed),
             "completed": i + 1, "max_model_copy_score_error": max_score_error},
            cache,
        )
        print(f"PROPLACE {ds}: {i + 1}/{len(source)}", flush=True)

    z = torch.stack(rows)
    failure = torch.tensor(failed, dtype=torch.bool)
    schema = schema_from_training(train, names) if ds == "german_credit" else None
    legality = torch.isfinite(z).all(1)
    if schema is not None:
        legality, _ = validate_cf(z, schema)
        failure |= ~legality
    failure |= torch.sigmoid(current(z)) < 0.5
    futures = future_models(ds, train.shape[1], future_root)
    favorable = train[valid(current, train)]
    lof = LocalOutlierFactor(
        n_neighbors=min(20, len(favorable) - 1), novelty=True
    ).fit(favorable.numpy())
    metrics = evaluate(z, source, current, futures, favorable, lof, failure)
    metrics.update(
        {"dataset": ds, "method": "PROPLACE", "legality_rate": float(legality.float().mean()),
         "model_copy_max_score_error": max_score_error}
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--future-root", required=True)
    parser.add_argument("--heloc-home", required=True)
    parser.add_argument("--diabetes-home", required=True)
    parser.add_argument("--german-home", required=True)
    parser.add_argument("--corrected-root", required=True)
    parser.add_argument("--margin-root", required=True)
    parser.add_argument("--logistic-root", required=True)
    parser.add_argument("--heloc-current-root", required=True)
    parser.add_argument("--diabetes-current-root", required=True)
    parser.add_argument("--german-root", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    roots = {
        "corrected": Path(args.corrected_root), "margin": Path(args.margin_root),
        "logistic": Path(args.logistic_root), "heloc_current": Path(args.heloc_current_root),
        "diabetes_current": Path(args.diabetes_current_root),
    }
    summaries = []
    for ds, home in (("heloc", args.heloc_home), ("diabetes", args.diabetes_home),
                     ("german_credit", args.german_home)):
        summaries.append(run_dataset(ds, home, roots, Path(args.german_root),
                                     Path(args.future_root), output))
    table = pd.DataFrame(summaries)
    table.to_csv(output / "proplace_results.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
