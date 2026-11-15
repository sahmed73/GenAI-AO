#!/usr/bin/env python3
"""Descriptor ANN with nested, chemistry-grouped cross-validation.

For each of the five outer folds:
  1. The four training folds are split into three inner folds.
  2. In each inner split, descriptors are scaled and ranked by ExtraTrees
     importance using the inner training molecules only. Every combination of
     descriptor count (64, 128, 256, 512) and ANN configuration is trained with
     early stopping on the inner validation fold.
  3. The combination with the highest mean inner validation R2 is selected; the
     number of training epochs is the median of its early-stopping epochs.
  4. Scaling and ranking are refitted on all four training folds, three
     independently initialized ANNs are trained for that number of epochs, and
     their mean prediction is evaluated on the held-out outer fold.
  5. SHAP values (expected gradients) are computed for the held-out molecules.

Run from the repository root:  python training/ann_train.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import ExtraTreesRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (DEFAULT_DATA, INNER_FOLDS, OUTER_FOLDS, TARGET, FoldTransformer,
                    assign_folds, group_bootstrap_interval, load_dataset, metrics,
                    seed_everything)
from model import DescriptorANN, Ensemble

DESCRIPTOR_COUNTS = (64, 128, 256, 512)
CONFIGURATIONS = [
    {"id": "mlp_00", "hidden_layers": [128, 64], "activation": "relu", "dropout": 0.10,
     "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 128},
    {"id": "mlp_01", "hidden_layers": [256, 128], "activation": "gelu", "dropout": 0.10,
     "learning_rate": 5e-4, "weight_decay": 1e-4, "batch_size": 128},
    {"id": "mlp_02", "hidden_layers": [256, 128, 64], "activation": "gelu", "dropout": 0.15,
     "learning_rate": 5e-4, "weight_decay": 1e-3, "batch_size": 128},
    {"id": "mlp_03", "hidden_layers": [128, 64], "activation": "gelu", "dropout": 0.20,
     "learning_rate": 1e-3, "weight_decay": 1e-3, "batch_size": 128},
    {"id": "mlp_04", "hidden_layers": [512, 256, 128], "activation": "gelu", "dropout": 0.20,
     "learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": 128},
]
CANDIDATES = [dict(c, id=f"{c['id']}_f{n}", descriptors=n) for c in CONFIGURATIONS for n in DESCRIPTOR_COUNTS]
MAX_EPOCHS, PATIENCE, MIN_DELTA, CLIP_NORM = 250, 25, 1e-5, 5.0
RANKING_TREES, MAX_RANKED = 256, 512
ENSEMBLE_SEEDS = (20260841, 20260842, 20260843)
SEED = 20260812
SHAP_BACKGROUND, SHAP_SAMPLES, SHAP_SEED = 100, 512, 20260926
BOOTSTRAP_REPLICATES = 2000


def rank_descriptors(x: np.ndarray, y: np.ndarray, seed: int, jobs: int) -> np.ndarray:
    """Column indices of the top descriptors by ExtraTrees importance."""
    trees = ExtraTreesRegressor(n_estimators=RANKING_TREES, min_samples_leaf=3, max_features="sqrt",
                                n_jobs=jobs, random_state=seed).fit(x, y)
    importance = trees.feature_importances_
    return np.lexsort((np.arange(len(importance)), -importance))[:MAX_RANKED]


def prepare(train, evaluate, descriptors, seed, jobs):
    """Fit scaling and ranking on `train`; return ranked matrices and names."""
    transformer = FoldTransformer(descriptors).fit(train)
    x_train, x_eval = transformer.transform(train), transformer.transform(evaluate)
    order = rank_descriptors(x_train, train[TARGET].to_numpy(float), seed, jobs)
    return x_train[:, order], x_eval[:, order], transformer.output_names_[order].tolist()


def _loader(x, y, batch_size, seed, device):
    tensors = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(tensors, batch_size=batch_size, shuffle=True, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda")


def _epoch(model, loader, optimizer, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()


def train_ann(x_train, y_train, config, seed, device, epochs=None, x_val=None, y_val=None):
    """Train on the standardized target. With validation data, early stopping
    returns the best epoch; otherwise the model is trained for `epochs`."""
    seed_everything(seed)
    mean, sd = float(np.mean(y_train)), float(np.std(y_train))
    model = DescriptorANN(x_train.shape[1], config["hidden_layers"], config["activation"],
                          config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    loader = _loader(x_train, (np.asarray(y_train) - mean) / sd, config["batch_size"], seed, device)
    if x_val is None:
        for _ in range(epochs):
            _epoch(model, loader, optimizer, device)
        return model.eval(), mean, sd, epochs
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor((np.asarray(y_val) - mean) / sd, dtype=torch.float32, device=device)
    best, best_epoch, best_state, stale = math.inf, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        _epoch(model, loader, optimizer, device)
        model.eval()
        with torch.no_grad():
            loss = float(nn.functional.mse_loss(model(xv), yv))
        if loss < best - MIN_DELTA:
            best, best_epoch, best_state, stale = loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.eval(), mean, sd, best_epoch


def predict(model, x, mean, sd, device) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy() * sd + mean


def run_outer_fold(data, folds, descriptors, outer, jobs, device, compute_shap):
    develop = data[folds.outer_fold != outer]
    test = data[folds.outer_fold == outer]
    inner_folds = folds.loc[develop.index, f"inner_fold_{outer}"].to_numpy()

    # Inner cross-validation: descriptor count and configuration.
    inner_rows = []
    for inner in range(INNER_FOLDS):
        train, valid = develop[inner_folds != inner], develop[inner_folds == inner]
        x_train, x_valid, _ = prepare(train, valid, descriptors, SEED + 1000 * outer + 10 * inner + 1, jobs)
        y_train, y_valid = train[TARGET].to_numpy(float), valid[TARGET].to_numpy(float)
        for index, config in enumerate(CANDIDATES):
            n = config["descriptors"]
            model, mean, sd, best_epoch = train_ann(
                x_train[:, :n], y_train, config, SEED + 10000 * outer + 100 * inner + index, device,
                x_val=x_valid[:, :n], y_val=y_valid)
            inner_rows.append({"outer_fold": outer, "inner_fold": inner, "candidate": config["id"],
                               "descriptors": n, "best_epoch": best_epoch,
                               **metrics(y_valid, predict(model, x_valid[:, :n], mean, sd, device))})
        print(f"outer {outer}: inner fold {inner} done", flush=True)
    inner_table = pd.DataFrame(inner_rows)
    summary = (inner_table.groupby("candidate", as_index=False)
               .agg(mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"), median_epoch=("best_epoch", "median"))
               .sort_values(["mean_r2", "mean_rmse", "candidate"], ascending=[False, True, True]))
    best = summary.iloc[0]
    config = next(c for c in CANDIDATES if c["id"] == best.candidate)
    epochs = max(1, int(round(float(best.median_epoch))))

    # Final models: refit on all four training folds, test on the held-out fold.
    n = config["descriptors"]
    x_dev, x_test, names = prepare(develop, test, descriptors, SEED + 1000 * outer + 777 + 1, jobs)
    x_dev, x_test, names = x_dev[:, :n], x_test[:, :n], names[:n]
    y_dev, y_test = develop[TARGET].to_numpy(float), test[TARGET].to_numpy(float)
    models, seed_predictions = [], []
    for i, base in enumerate(ENSEMBLE_SEEDS):
        model, mean, sd, _ = train_ann(x_dev, y_dev, config, base + 100 * outer + i, device, epochs=epochs)
        models.append(model)
        seed_predictions.append(predict(model, x_test, mean, sd, device))
    prediction = np.mean(seed_predictions, axis=0)

    predictions = pd.DataFrame({"smiles": test.smiles.to_numpy(), "chemistry_group": folds.loc[test.index, "chemistry_group"].to_numpy(),
                                "outer_fold": outer, "k_simulated": y_test, "k_predicted": prediction,
                                **{f"k_predicted_seed_{i + 1}": p for i, p in enumerate(seed_predictions)}})
    selected = {"outer_fold": outer, "configuration": config["id"].rsplit("_f", 1)[0],
                "descriptors": n, "epochs": epochs, "mean_inner_r2": float(best.mean_r2),
                **{k: v for k, v in config.items() if k not in ("id", "descriptors")}}
    features = pd.DataFrame({"outer_fold": outer, "rank": np.arange(1, n + 1), "descriptor": names})

    shap_table = None
    if compute_shap:
        ensemble = Ensemble(models).eval()
        rng = np.random.default_rng(SHAP_SEED + outer)
        background = torch.as_tensor(x_dev[rng.choice(len(x_dev), SHAP_BACKGROUND, replace=False)],
                                     dtype=torch.float32, device=device)
        explainer = shap.GradientExplainer(ensemble, background, batch_size=256)
        values = np.asarray(explainer.shap_values(torch.as_tensor(x_test, dtype=torch.float32, device=device),
                                                  nsamples=SHAP_SAMPLES, rseed=SHAP_SEED + outer))
        values = values.reshape(len(test), n) * sd
        raw = test[names].to_numpy(float)
        shap_table = pd.DataFrame({
            "smiles": np.repeat(test.smiles.to_numpy(), n),
            "outer_fold": outer,
            "descriptor": np.tile(names, len(test)),
            "descriptor_value": raw.ravel(),
            "shap_value": values.ravel(),
        })
    return inner_table, predictions, selected, features, shap_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results/ann"))
    parser.add_argument("--outer-folds", type=int, nargs="+", default=list(range(OUTER_FOLDS)))
    parser.add_argument("--jobs", type=int, default=8, help="CPU threads for ExtraTrees ranking")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.jobs)
    data, descriptors = load_dataset(args.data)
    folds = assign_folds(data)
    args.output.mkdir(parents=True, exist_ok=True)
    folds.assign(smiles=data.smiles).to_csv(args.output / "folds.csv", index=False)
    print(f"{len(data)} molecules, {len(descriptors)} candidate descriptors, device {device}", flush=True)

    for outer in args.outer_folds:
        inner, predictions, selected, features, shap_table = run_outer_fold(
            data, folds, descriptors, outer, args.jobs, device, not args.no_shap)
        stem = args.output / f"outer_fold_{outer}"
        inner.to_csv(f"{stem}_inner_validation.csv", index=False)
        predictions.to_csv(f"{stem}_predictions.csv", index=False)
        features.to_csv(f"{stem}_descriptors.csv", index=False)
        Path(f"{stem}_selection.json").write_text(json.dumps(selected, indent=2) + "\n")
        if shap_table is not None:
            shap_table.to_csv(f"{stem}_shap.csv.gz", index=False)
        print(f"outer {outer}: R2 = {metrics(predictions.k_simulated, predictions.k_predicted)['r2']:.4f}", flush=True)

    if sorted(args.outer_folds) == list(range(OUTER_FOLDS)):
        summarize(args.output)


def summarize(output: Path) -> None:
    """Pooled held-out metrics, per-fold metrics and SHAP importance."""
    predictions = pd.concat([pd.read_csv(output / f"outer_fold_{f}_predictions.csv") for f in range(OUTER_FOLDS)],
                            ignore_index=True)
    predictions.to_csv(output / "held_out_predictions.csv", index=False)
    y, p = predictions.k_simulated.to_numpy(), predictions.k_predicted.to_numpy()
    pooled = metrics(y, p)
    low, high = group_bootstrap_interval(np.arange(len(y)), predictions.chemistry_group,
                                         lambda i: metrics(y[i], p[i])["r2"],
                                         BOOTSTRAP_REPLICATES, SEED + 9000)
    per_fold = [{"outer_fold": f, **metrics(g.k_simulated, g.k_predicted)} for f, g in predictions.groupby("outer_fold")]
    selections = [json.loads((output / f"outer_fold_{f}_selection.json").read_text()) for f in range(OUTER_FOLDS)]
    pd.DataFrame(per_fold).to_csv(output / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(output / "selected_configurations.csv", index=False)
    result = {"molecules": len(predictions), **{f"pooled_{k}": v for k, v in pooled.items()},
              "pooled_r2_95ci": [low, high]}
    shap_files = [output / f"outer_fold_{f}_shap.csv.gz" for f in range(OUTER_FOLDS)]
    if all(f.exists() for f in shap_files):
        values = pd.concat([pd.read_csv(f) for f in shap_files], ignore_index=True)
        common = set.intersection(*(set(g.descriptor) for _, g in values.groupby("outer_fold")))
        importance = (values[values.descriptor.isin(common)].assign(abs_shap=lambda v: v.shap_value.abs())
                      .groupby("descriptor", as_index=False).abs_shap.mean()
                      .rename(columns={"abs_shap": "mean_abs_shap"})
                      .sort_values("mean_abs_shap", ascending=False))
        importance.to_csv(output / "shap_importance.csv", index=False)
        result["descriptors_selected_in_all_folds"] = len(common)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
