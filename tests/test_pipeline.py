"""Fast end-to-end and unit tests. Run with: python -m pytest tests -q"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automl.io import load_dataset  # noqa: E402
from automl.pipeline import run  # noqa: E402
from automl.preprocess import Preprocessor, analyse_columns  # noqa: E402
from automl.profile import infer_column_types  # noqa: E402

rng = np.random.default_rng(0)


def _messy_frame(n=300):
    df = pd.DataFrame({
        "id": np.arange(n),
        "num": rng.normal(size=n),
        "num_str": [f"{v:,.2f}" for v in rng.normal(size=n) * 1000],
        "big": rng.normal(size=n) * 1e6,
        "cat": rng.choice(["a", "b", "c"], n),
        "cat_dup": None,
        "const": 5,
        "date": pd.date_range("2021-01-01", periods=n).strftime("%Y-%m-%d"),
        "bools": rng.choice([True, False], n),
    })
    df.loc[rng.random(n) < 0.1, "num"] = np.nan
    df.loc[rng.random(n) < 0.1, "cat"] = None
    df["cat_dup"] = df["cat"].map({"a": 1, "b": 2, "c": 3})   # relabelled duplicate
    df["num_dup"] = df["num"]                                  # exact duplicate
    return df


def test_column_inference_and_drops():
    df = _messy_frame()
    types = infer_column_types(df)
    assert types["num"] == "numeric" and types["num_str"] == "numeric"
    assert types["cat"] == "categorical" and types["bools"] == "categorical"
    assert types["date"] == "datetime"
    _, drop = analyse_columns(df)
    assert "const" in drop["constant"]
    assert "id" in drop["id_like"]
    assert drop["duplicate"].get("num_dup") == "num"
    assert drop["duplicate"].get("cat_dup") == "cat"
    assert "big" not in drop["dropped"] and "num_str" not in drop["dropped"]


def test_preprocessor_handles_unseen_values_and_nans():
    df = _messy_frame()
    for mode in ("tree", "linear"):
        pre = Preprocessor(mode=mode).fit(df)
        M = pre.transform(df)
        assert M.shape[0] == len(df) and np.isfinite(M).all()
        new = df.head(5).copy()
        new["cat"] = "never_seen"
        new.loc[:, "num"] = np.nan
        M2 = pre.transform(new)
        assert M2.shape[1] == M.shape[1] and np.isfinite(M2).all()


def _run(tmp_path, train, test, meta):
    d = tmp_path / "ds"
    d.mkdir()
    train.to_csv(d / "train.csv", index=False)
    if test is not None:
        test.to_csv(d / "test.csv", index=False)
    (d / "metadata.json").write_text(json.dumps(meta))
    ds = load_dataset(str(d))
    out = tmp_path / "out"
    rep = run(ds, str(out), time_limit=20, verbose=False)
    assert "error" not in rep, rep
    preds = pd.read_csv(out / "predictions.csv")
    prof = json.loads((out / "profile.json").read_text())
    return rep, preds, prof


def test_classification_end_to_end(tmp_path):
    df = _messy_frame(400)
    df["target"] = np.where(df["num"].fillna(0) + df["cat_dup"] > 2, "yes", "no")
    rep, preds, prof = _run(tmp_path, df.iloc[:300], df.iloc[300:].drop(columns="target"),
                            {"task": "classification", "target": "target", "id_column": "id"})
    assert len(preds) == 100 and set(preds["prediction"]) <= {"yes", "no"}
    assert list(preds.columns) == ["id", "prediction"]
    assert prof["task"] == "classification" and prof["target_info"]["n_classes"] == 2
    assert "const" in prof["summary"]["constant_columns"]
    assert rep["cv_score"] > 0.8


def test_regression_end_to_end(tmp_path):
    df = _messy_frame(400)
    df["y"] = 3 * df["num"].fillna(0) + df["cat_dup"] * 10 + rng.normal(size=400)
    rep, preds, prof = _run(tmp_path, df.iloc[:300], df.iloc[300:].drop(columns="y"),
                            {"task": "regression", "target": "y"})
    assert len(preds) == 100 and np.isfinite(preds["prediction"]).all()
    assert prof["target_info"]["name"] == "y"


def test_clustering_end_to_end(tmp_path):
    centers = np.array([[0, 0], [10, 10], [-10, 10]])
    lab = rng.integers(0, 3, 300)
    X = centers[lab] + rng.normal(size=(300, 2))
    df = pd.DataFrame(X, columns=["a", "b"])
    df["noise"] = rng.normal(size=300)
    df["colour"] = rng.choice(["red", "blue"], 300)
    rep, preds, prof = _run(tmp_path, df, None, {"task": "clustering", "n_clusters": 3})
    assert len(preds) == 300 and preds["prediction"].nunique() == 3
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(lab, preds["prediction"]) > 0.9


def test_tiny_and_single_feature(tmp_path):
    df = pd.DataFrame({"x": rng.normal(size=30)})
    df["y"] = (df["x"] > 0).astype(int)
    rep, preds, _ = _run(tmp_path, df.iloc[:20], df.iloc[20:].drop(columns="y"),
                         {"task": "classification", "target": "y"})
    assert len(preds) == 10


def test_metadata_variants(tmp_path):
    df = _messy_frame(100)
    df["label"] = rng.choice([0, 1], 100)
    d = tmp_path / "ds"
    d.mkdir()
    df.to_csv(d / "train.csv", index=False)
    (d / "metadata.json").write_text(json.dumps({
        "task_type": "binary", "label_column": "label",
        "feature_types": {"cat_dup": "categorical", "num": "numeric"}}))
    ds = load_dataset(str(d))
    assert ds.task == "classification" and ds.target == "label"
    assert ds.feature_types["cat_dup"] == "categorical"


def test_id_detection_is_conservative():
    n = 100
    df = pd.DataFrame({
        "age_rank": np.arange(1, n + 1),            # consecutive but not first column
        "customer_id": rng.permutation(n) + 5000,   # named id, unique
        "score": rng.normal(size=n),
        "token": [f"{i:032x}" for i in rng.permutation(n)],  # hash-like strings
    })
    df.insert(0, "idx", np.arange(n))               # consecutive, first column
    _, drop = analyse_columns(df)
    assert set(drop["id_like"]) == {"idx", "customer_id", "token"}
    assert "age_rank" not in drop["dropped"]


def test_declared_id_column_is_respected(tmp_path):
    n = 120
    df = pd.DataFrame({"x": rng.normal(size=n), "ref": rng.permutation(n) + 10})
    df["y"] = (df["x"] > 0).astype(int)
    rep, preds, prof = _run(tmp_path, df.iloc[:90], df.iloc[90:].drop(columns="y"),
                            {"task": "classification", "target": "y", "id": "ref"})
    assert list(preds.columns) == ["ref", "prediction"]
    assert "ref" in prof["summary"]["id_like_columns"]


def test_single_class_and_rare_class(tmp_path):
    df = pd.DataFrame({"x": rng.normal(size=60), "y": 1})
    rep, preds, _ = _run(tmp_path, df.iloc[:40], df.iloc[40:].drop(columns="y"),
                         {"task": "classification", "target": "y"})
    assert (preds["prediction"] == 1).all()
    df["y"] = 0
    df.loc[3, "y"] = 1                               # one positive example only
    (tmp_path / "b").mkdir()
    rep, preds, _ = _run(tmp_path / "b", df.iloc[:40], df.iloc[40:].drop(columns="y"),
                         {"task": "classification", "target": "y"})
    assert len(preds) == 20


def test_one_to_three_training_rows(tmp_path):
    for n in (1, 2, 3):
        df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n) * 2.0})
        test = pd.DataFrame({"x": [1.0, 2.0]})
        d = tmp_path / f"n{n}"
        d.mkdir()
        rep, preds, _ = _run(d, df, test, {"task": "regression", "target": "y"})
        assert len(preds) == 2 and np.isfinite(preds["prediction"]).all()
    df = pd.DataFrame({"x": [0.0, 1.0], "y": ["a", "b"]})
    (tmp_path / "c").mkdir()
    rep, preds, _ = _run(tmp_path / "c", df, test, {"task": "classification", "target": "y"})
    assert len(preds) == 2


def test_too_few_rows_for_cv(tmp_path):
    df = pd.DataFrame({"x": rng.normal(size=12)})
    df["y"] = 2 * df["x"]
    rep, preds, _ = _run(tmp_path, df.iloc[:6], df.iloc[6:].drop(columns="y"),
                         {"task": "regression", "target": "y"})
    assert rep.get("cv_folds") == 0 and len(preds) == 6


def test_watchdog_keeps_fallback_outputs(tmp_path):
    """A hard time limit terminates the run with valid fallback files on disk."""
    import subprocess
    n = 6000
    X = rng.normal(size=(n, 40))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(40)])
    df["y"] = (X[:, :5].sum(1) > 0).astype(int)
    d = tmp_path / "ds"
    d.mkdir()
    df.to_csv(d / "train.csv", index=False)
    (d / "metadata.json").write_text(json.dumps({"task": "classification", "target": "y"}))
    out = tmp_path / "out"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    t0 = __import__("time").time()
    subprocess.run([sys.executable, os.path.join(root, "run.py"), str(d), "--out", str(out),
                    "--time-limit", "6", "--quiet"], timeout=20)
    assert __import__("time").time() - t0 < 8
    assert (out / "predictions.csv").exists() and (out / "profile.json").exists()
    assert len(pd.read_csv(out / "predictions.csv")) == n
