#!/usr/bin/env python3
"""Generate messy synthetic datasets for local testing.

Each dataset gets train.csv, test.csv, metadata.json and (hidden) solution.csv.
The mess: missing values, mixed scales, categoricals (as strings, with rare
levels), constant and duplicate columns, irrelevant noise features, class
imbalance, nonlinear targets, id columns, numbers stored as strings.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_friedman1

RS = 7


def _mess(X: pd.DataFrame, rng, missing=0.08, n_noise=4, n_cat=3, n_const=1, n_dup=1,
          scale=True, string_numbers=True):
    n = len(X)
    for i in range(n_noise):
        X[f"noise_{i}"] = rng.normal(size=n)
    for i in range(n_cat):
        levels = [f"L{j}" for j in range(rng.integers(3, 8))] + ["rare"]
        p = rng.dirichlet(np.ones(len(levels)))
        p[-1] = 0.005
        p /= p.sum()
        X[f"cat_noise_{i}"] = rng.choice(levels, size=n, p=p)
    for i in range(n_const):
        X[f"const_{i}"] = 1.0
    cols = [c for c in X.columns if c.startswith("f")]
    if scale:
        for c in cols[::2]:
            X[c] = X[c] * rng.choice([1e-3, 1, 1e3, 1e5])
    # Missing values in most columns.
    for c in X.columns:
        if c.startswith("const"):
            continue
        m = rng.random(n) < missing * rng.random() * 2
        X.loc[m, c] = np.nan
    for i in range(n_dup):  # exact copies, added after missingness
        X[f"dup_{i}"] = X[cols[i % len(cols)]].values
    if string_numbers and cols:
        c = cols[-1]
        X[c] = X[c].map(lambda v: "" if pd.isna(v) else f"{v:,.3f}")
    return X


def _finish(name, X, y, meta, out, rng, test_frac=0.3, id_col=True):
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    X = X.copy()
    if id_col:
        X.insert(0, "row_id", np.arange(1000, 1000 + n))
        meta["id_column"] = "row_id"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    train = X.iloc[tr].copy()
    test = X.iloc[te].copy()
    if y is not None and test_frac > 0:
        train[meta["target"]] = np.asarray(y)[tr]
        sol = pd.DataFrame({meta["target"]: np.asarray(y)[te]})
        if id_col:
            sol.insert(0, "row_id", test["row_id"].values)
        sol.to_csv(os.path.join(d, "solution.csv"), index=False)
    elif y is not None:  # clustering: hidden labels aligned with the train rows
        sol = pd.DataFrame({"row_id": train["row_id"].values, "cluster": np.asarray(y)[tr]})
        sol.to_csv(os.path.join(d, "solution.csv"), index=False)
    train.to_csv(os.path.join(d, "train.csv"), index=False)
    if len(test):
        test.to_csv(os.path.join(d, "test.csv"), index=False)
    with open(os.path.join(d, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {d}: train={train.shape} test={test.shape}")


def main(out="data/synthetic"):
    rng = np.random.default_rng(RS)
    # 1. imbalanced binary classification with categorical signal
    Xa, ya = make_classification(4000, 12, n_informative=6, n_redundant=2, weights=[0.92],
                                 flip_y=0.02, class_sep=1.0, random_state=RS)
    X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(12)])
    sig_cat = np.where(ya == 1, rng.choice(["A", "B"], 4000, p=[.7, .3]),
                       rng.choice(["A", "B", "C"], 4000, p=[.2, .3, .5]))
    X["plan"] = sig_cat
    X = _mess(X, rng)
    _finish("clf_imbalanced", X, np.where(ya == 1, "yes", "no"),
            {"task": "classification", "target": "churn"}, out, rng)

    # 2. multiclass with nonlinear boundaries, no id column
    Xb, yb = make_classification(3000, 10, n_informative=5, n_classes=4, n_clusters_per_class=2,
                                 class_sep=1.3, random_state=RS + 1)
    X = pd.DataFrame(Xb, columns=[f"f{i}" for i in range(10)])
    X = _mess(X, rng, n_cat=2)
    _finish("clf_multiclass", X, yb, {"task": "classification", "target": "label"},
            out, rng, id_col=False)

    # 3. nonlinear regression (Friedman) with a skewed target and a date column
    Xc, yc = make_friedman1(3500, 10, noise=0.5, random_state=RS)
    X = pd.DataFrame(Xc, columns=[f"f{i}" for i in range(10)])
    y = np.exp(yc / 6) * 1000
    X["signup_date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 1500, 3500), unit="D")
    X["signup_date"] = X["signup_date"].dt.strftime("%Y-%m-%d")
    X = _mess(X, rng)
    _finish("reg_nonlinear_skewed", X, y, {"task": "regression", "target": "price"}, out, rng)

    # 4. mostly linear regression with categorical effects and heavy missingness
    n = 5000
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(8)})
    city = rng.choice(["Paris", "Lyon", "Nice", "Lille", "Metz"], n)
    eff = pd.Series(city).map({"Paris": 50, "Lyon": 20, "Nice": 10, "Lille": -5, "Metz": -20})
    y = 3 * X.f0 - 2 * X.f1 + 0.5 * X.f2 * X.f3 + eff.values + rng.normal(scale=2, size=n)
    X["city"] = city
    X = _mess(X, rng, missing=0.2, n_dup=2)
    _finish("reg_linear_missing", X, y, {"task": "regression", "target": "y"}, out, rng)

    # 5. clustering: 4 blobs in 3 informative dims + 6 noise dims, mixed scales
    Xd, yd = make_blobs(2500, 3, centers=4, cluster_std=1.0, random_state=RS)
    X = pd.DataFrame(Xd, columns=["income", "spend", "age"])
    X["income"] *= 1e4
    for i in range(6):
        X[f"noise_{i}"] = rng.normal(scale=3, size=2500)
    X["const"] = 0
    X["segment_dup"] = X["spend"].values
    _finish("clu_blobs_noise", X, yd,
            {"task": "clustering", "n_clusters": 4, "hidden_labels": "solution"}, out, rng,
            test_frac=0.0)

    # 6. clustering: 3 anisotropic / unequal-size groups with a categorical column
    rng2 = np.random.default_rng(RS + 5)
    parts = [rng2.multivariate_normal([0, 0], [[4, 3.5], [3.5, 4]], 900),
             rng2.multivariate_normal([8, -6], [[1, 0], [0, 1]], 300),
             rng2.multivariate_normal([-7, 7], [[2, -1], [-1, 2]], 600)]
    Xe = np.vstack(parts)
    ye = np.repeat([0, 1, 2], [900, 300, 600])
    X = pd.DataFrame(Xe, columns=["x1", "x2"])
    X["x1"] = X["x1"] * 100
    X["group_hint"] = np.where(ye == 1, "small", rng2.choice(["big", "mid"], len(ye)))
    X["noise"] = rng2.normal(size=len(ye))
    _finish("clu_aniso", X, ye, {"task": "clustering", "n_clusters": 3}, out, rng,
            test_frac=0.0)

    # 7. larger classification to test the time budget
    Xf, yf = make_classification(60000, 30, n_informative=12, n_classes=3, random_state=RS + 2)
    X = pd.DataFrame(Xf, columns=[f"f{i}" for i in range(30)])
    X = _mess(X, rng, n_noise=10, n_cat=4, string_numbers=False)
    _finish("clf_large", X, yf, {"task": "classification", "target": "target"}, out, rng)

    # 8. larger regression
    Xg, yg = make_friedman1(50000, 20, noise=1.0, random_state=RS + 3)
    X = pd.DataFrame(Xg, columns=[f"f{i}" for i in range(20)])
    X = _mess(X, rng, n_noise=8, string_numbers=False)
    _finish("reg_large", X, yg, {"task": "regression", "target": "target"}, out, rng)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
