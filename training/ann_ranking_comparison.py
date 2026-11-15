#!/usr/bin/env python3
"""Compare descriptor-ranking methods on the inner folds.

Rankings: LASSO path, mutual information, minimum-redundancy maximum-relevance
(mRMR) and ExtraTrees importance. Each ranking is fitted on the inner training
molecules; the top 32, 64, 128 or 256 descriptors are evaluated on the inner
validation fold with a Ridge (linear) and an ExtraTrees (nonlinear) regressor.
ann_train.py uses the ExtraTrees ranking.

Run from the repository root:  python training/ann_ranking_comparison.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge, lasso_path

from common import DEFAULT_DATA, INNER_FOLDS, OUTER_FOLDS, TARGET, FoldTransformer, assign_folds, load_dataset, metrics

COUNTS = (32, 64, 128, 256)
SEED = 20260804


def descending(scores) -> np.ndarray:
    scores = np.nan_to_num(np.asarray(scores, float), nan=-np.inf)
    return np.lexsort((np.arange(len(scores)), -scores))


def lasso_ranking(x, y):
    """Fraction of the regularization path on which a descriptor is nonzero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, coef, _ = lasso_path(x.astype(float), (y - y.mean()) / y.std(), eps=1e-3, n_alphas=60,
                                max_iter=10000, tol=1e-5)
    size = np.abs(coef).max(axis=1)
    return descending((np.abs(coef) > 1e-10).mean(axis=1) + 1e-3 * size / max(size.max(), 1e-12))


def mrmr_ranking(x, relevance, count):
    """Greedy mutual-information relevance minus mean absolute correlation."""
    values = x.astype(np.float32)
    correlation = np.abs(values.T @ values / len(values))
    np.fill_diagonal(correlation, 0.0)
    relevance = relevance / max(relevance.max(), 1e-12)
    chosen, available, redundancy = [], np.ones(x.shape[1], bool), np.zeros(x.shape[1])
    for step in range(count):
        merit = relevance - (redundancy / step if step else 0.0)
        merit[~available] = -np.inf
        best = int(np.argmax(merit))
        chosen.append(best)
        available[best] = False
        redundancy += correlation[:, best]
    rest = [i for i in descending(relevance) if available[i]]
    return np.asarray(chosen + rest)


def rankings(x, y, seed, jobs):
    mi = np.nan_to_num(mutual_info_regression(x, y, n_neighbors=3, random_state=seed))
    trees = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=3, max_features="sqrt",
                                n_jobs=jobs, random_state=seed + 1).fit(x, y)
    return {"lasso": lasso_ranking(x, y), "mutual_information": descending(mi),
            "mrmr": mrmr_ranking(x, mi, max(COUNTS)), "extra_trees": descending(trees.feature_importances_)}


def probe(kind, x_train, y_train, x_valid, seed, jobs):
    mean, sd = y_train.mean(), y_train.std()
    model = (Ridge(alpha=10.0) if kind == "ridge" else
             ExtraTreesRegressor(n_estimators=96, min_samples_leaf=3, max_features=0.7,
                                 n_jobs=jobs, random_state=seed))
    return model.fit(x_train, (y_train - mean) / sd).predict(x_valid) * sd + mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("results/ann_ranking_comparison"))
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    data, descriptors = load_dataset(args.data)
    folds = assign_folds(data)
    rows = []
    for outer in range(OUTER_FOLDS):
        develop = data[folds.outer_fold != outer]
        inner_folds = folds.loc[develop.index, f"inner_fold_{outer}"].to_numpy()
        for inner in range(INNER_FOLDS):
            train, valid = develop[inner_folds != inner], develop[inner_folds == inner]
            transformer = FoldTransformer(descriptors).fit(train)
            x_train, x_valid = transformer.transform(train), transformer.transform(valid)
            y_train, y_valid = train[TARGET].to_numpy(float), valid[TARGET].to_numpy(float)
            seed = SEED + 100 * outer + 10 * inner
            for method, order in rankings(x_train, y_train, seed, args.jobs).items():
                for count in COUNTS:
                    for p, kind in enumerate(("ridge", "extra_trees")):
                        pred = probe(kind, x_train[:, order[:count]], y_train, x_valid[:, order[:count]],
                                     seed + 1000 + p, args.jobs)
                        rows.append({"outer_fold": outer, "inner_fold": inner, "method": method,
                                     "descriptors": count, "regressor": kind, **metrics(y_valid, pred)})
            print(f"outer {outer} inner {inner} done", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "inner_validation.csv", index=False)
    summary = table.groupby(["method", "regressor"], as_index=False).r2.mean().rename(columns={"r2": "mean_r2"})
    overall = table.groupby("method", as_index=False).r2.mean().rename(columns={"r2": "mean_r2"})
    summary.to_csv(args.output / "summary_by_regressor.csv", index=False)
    overall.sort_values("mean_r2", ascending=False).to_csv(args.output / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(overall.sort_values("mean_r2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
