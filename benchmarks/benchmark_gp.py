# benchmarks/benchmark_gp.py
"""
SmallGP benchmark — normalized MAE, median, rank.

Normalization: MAE / std(y) to make datasets comparable.

Models: SmallGP, SmallMLP, SmallGBM, XGBoost, LightGBM, RF, MLP, KNN.

Run:
    python3 benchmarks/benchmark_gp.py
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import (
    load_diabetes,
    make_friedman1, make_friedman2, make_friedman3,
    make_regression,
)

from smallgp import SmallGPRegressor

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

def build_datasets():
    datasets = []

    diab = load_diabetes()
    datasets.append(("diabetes", diab.data, diab.target))
    rng = np.random.default_rng(7)
    for n_sub in [50, 100, 200]:
        idx = rng.choice(len(diab.target), size=n_sub, replace=False)
        datasets.append((f"diabetes_{n_sub}", diab.data[idx], diab.target[idx]))

    for n in [100, 300, 500]:
        for name, fn in [
            ("friedman1", make_friedman1),
            ("friedman2", make_friedman2),
            ("friedman3", make_friedman3),
        ]:
            try:
                X, y = fn(n_samples=n, noise=0.1, random_state=42)
                datasets.append((f"{name}_{n}", X, y))
            except Exception:
                pass

    for n, d, n_inf in [(100, 20, 5), (300, 30, 8), (500, 40, 10)]:
        X, y = make_regression(
            n_samples=n, n_features=d, n_informative=n_inf,
            noise=0.3, random_state=42,
        )
        datasets.append((f"mreg_{n}x{d}_inf{n_inf}", X, y))

    configs = [
        (80, 5, "sin"), (100, 10, "sin"), (150, 10, "sin"),
        (200, 20, "sin"), (300, 15, "poly"), (500, 20, "poly"),
    ]
    for i, (n, d, kind) in enumerate(configs):
        rng = np.random.default_rng(100 + i)
        X = rng.normal(size=(n, d))
        if kind == "sin":
            y = np.sin(3 * X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)
        else:
            y = X[:, 0] ** 2 - X[:, 1] * X[:, 2] + 0.1 * rng.normal(size=n)
        datasets.append((f"synth_{kind}_{n}x{d}", X, y))

    return datasets


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

def build_models():
    models = {
        "SmallGP": SmallGPRegressor(),
        "MLP_100": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(100,),
                                    max_iter=1000, random_state=42)),
        ]),
        "MLP_wide": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(256, 128),
                                    max_iter=1000, random_state=42)),
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "RF_100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    try:
        from smallmlp import SmallMLPRegressor
        models["SmallMLP"] = SmallMLPRegressor(max_epochs=500, patience=30)
    except ImportError:
        pass

    try:
        from smallgbm import SmallGBMRegressor
        models["SmallGBM"] = SmallGBMRegressor()
    except ImportError:
        pass

    # try:
    #     from xgboost import XGBRegressor
    #     models["XGBoost"] = XGBRegressor(
    #         n_estimators=200, max_depth=3, learning_rate=0.1,
    #         random_state=42, verbosity=0,
    #     )
    # except ImportError:
    #     pass

    # try:
    #     from lightgbm import LGBMRegressor
    #     models["LightGBM"] = LGBMRegressor(
    #         n_estimators=200, max_depth=3, learning_rate=0.1,
    #         random_state=42, verbose=-1,
    #     )
    # except ImportError:
    #     pass

    return models


# ----------------------------------------------------------------------
# Benchmark — normalized MAE
# ----------------------------------------------------------------------

def run_benchmark(n_splits=5, seed=42):
    datasets = build_datasets()
    models = build_models()
    print(f"\nDatasets: {len(datasets)}  Models: {len(models)}")
    print("Metric: normalized MAE = MAE / std(y)")
    print("=" * 110)

    rows = []
    for name, X, y in datasets:
        if len(y) > 500:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(y), 500, replace=False)
            X, y = X[idx], y[idx]

        y_std = float(np.std(y))
        if y_std < 1e-9:
            y_std = 1.0

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        maes = {m: [] for m in models}

        for tr, te in kf.split(X):
            for mname, model in models.items():
                try:
                    m = clone(model)
                    m.fit(X[tr], y[tr])
                    y_pred = m.predict(X[te])
                    raw = mean_absolute_error(y[te], y_pred)
                    maes[mname].append(raw / y_std)
                except Exception as e:
                    warnings.warn(f"{name} ({mname}): {e}")
                    maes[mname].append(np.nan)

        row = {"dataset": name, "n": len(y), "d": X.shape[1], "y_std": y_std}
        for m in models:
            row[m] = float(np.nanmean(maes[m]))
        rows.append(row)

        parts = [f"{m}={row[m]:.3f}" for m in models]
        print(f"{name:<28} " + "  ".join(parts), flush=True)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def summarize(df):
    model_cols = [c for c in df.columns if c not in ("dataset", "n", "d", "y_std")]
    pivot = df.set_index("dataset")[model_cols]

    print("\n" + "=" * 110)
    print("Normalized MAE (lower = better)")
    print("=" * 110)
    print(f"{'model':<12} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10}")
    for m in model_cols:
        vals = pivot[m].values
        print(f"{m:<12} {np.mean(vals):>10.4f} {np.median(vals):>10.4f} "
              f"{np.std(vals):>10.4f} {np.min(vals):>10.4f} {np.max(vals):>10.4f}")

    print("\n" + "=" * 110)
    print("Mean rank (lower = better)")
    print("=" * 110)
    ranks = pivot.rank(axis=1, method="average")
    for m in ranks.mean(axis=0).sort_values().index:
        print(f"{m:<12} {ranks[m].mean():>10.3f}")

    print("\n" + "=" * 110)
    print("Win count (best on dataset)")
    print("=" * 110)
    wins = (pivot == pivot.min(axis=1).values[:, None]).sum(axis=0).sort_values(ascending=False)
    for m, w in wins.items():
        print(f"{m:<12} {int(w):>6} / {len(pivot)}")

    print("\n" + "=" * 110)
    print("Top-3 count")
    print("=" * 110)
    top3 = np.zeros(len(model_cols))
    for ds in pivot.index:
        row = pivot.loc[ds].values
        order = np.argsort(row)
        for j in order[:3]:
            top3[j] += 1
    top3_s = pd.Series(top3, index=model_cols).sort_values(ascending=False)
    for m, c in top3_s.items():
        print(f"{m:<12} {int(c):>6} / {len(pivot)}")

    print("\n" + "=" * 110)
    print("Wilcoxon vs SmallGP (paired across datasets)")
    print("=" * 110)
    from scipy.stats import wilcoxon
    if "SmallGP" in pivot.columns:
        sm = pivot["SmallGP"].values
        print(f"{'baseline':<12} {'stat':>10} {'p':>10} {'wins':>6} {'losses':>8}")
        for m in model_cols:
            if m == "SmallGP":
                continue
            other = pivot[m].values
            wins = int(np.sum(sm < other - 1e-9))
            losses = int(np.sum(sm > other + 1e-9))
            try:
                stat, p = wilcoxon(sm, other, zero_method="wilcox", alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan
            print(f"{m:<12} {stat:>10.3f} {p:>10.4f} {wins:>6} {losses:>8}")


if __name__ == "__main__":
    df = run_benchmark(n_splits=5, seed=42)
    df.to_csv("benchmarks/results_gp.csv", index=False)
    summarize(df)
    print("\nSaved to benchmarks/results_gp.csv")