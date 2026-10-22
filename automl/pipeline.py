"""End-to-end run for one dataset: load -> profile -> model -> write outputs."""
from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import pandas as pd

from . import config
from .clustering import cluster
from .io import Dataset, load_dataset, write_json, write_predictions
from .preprocess import Preprocessor, analyse_columns
from .profile import build_profile
from .supervised import SupervisedAutoML


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def run(ds: Dataset, out_dir: str, time_limit: float | None = None, verbose=True) -> dict:
    t_start = time.time()
    log = _log if verbose else (lambda m: None)
    time_limit = time_limit or config.TIME_LIMIT_SECONDS
    deadline = t_start + time_limit - config.SAFETY_MARGIN_SECONDS
    log(f"task={ds.task} target={ds.target} k={ds.n_clusters} "
        f"train={ds.train.shape} test={None if ds.test is None else ds.test.shape}")

    feats = [c for c in ds.train.columns if c != ds.target]
    X = ds.train[feats]
    # The id column is passed through to the output but never used as a feature.
    col_types, drop_info = analyse_columns(X, ds.feature_types)
    if ds.id_col and ds.id_col in X.columns and ds.id_col not in drop_info["dropped"]:
        drop_info["id_like"].append(ds.id_col)
        drop_info["dropped"].append(ds.id_col)
    log(f"dropped: {drop_info['dropped']}")

    # 1. profile first so it is on disk even if modelling times out.
    profile = build_profile(ds, col_types, drop_info)
    write_json(profile, out_dir, config.PROFILE_FILE)
    log(f"profile written ({time.time() - t_start:.1f}s)")

    # 2. model
    X_pred = ds.test[feats] if ds.test is not None and all(
        c in ds.test.columns for c in feats) else (
        ds.test if ds.test is not None else ds.train[feats])
    if ds.test is not None and not all(c in ds.test.columns for c in feats):
        missing = [c for c in feats if c not in ds.test.columns]
        log(f"warning: test is missing columns {missing}; filling with NaN")
        X_pred = ds.test.reindex(columns=feats)
    ids = None
    if ds.id_col:
        src = ds.test if ds.test is not None else ds.train
        if ds.id_col in src.columns:
            ids = src[ds.id_col].to_numpy()

    report: dict = {"task": ds.task}
    try:
        if ds.task in ("classification", "regression"):
            # Fast fallback first: guarantees a valid prediction file quickly.
            y = ds.train[ds.target]
            _write_fallback(ds, y, X_pred, ids, out_dir)
            aml = SupervisedAutoML(ds.task, deadline, col_types, drop_info,
                                   ds.feature_types, log=log)
            aml.fit(X, y)
            preds = aml.predict(X_pred)
            report.update(aml.report)
        else:
            pre = Preprocessor(mode="linear", hints=ds.feature_types,
                               col_types=col_types, drop_info=drop_info, max_onehot=50)
            Xm = pre.fit_transform(X)
            if ds.test is not None:
                # Label the test rows; train rows only define the preprocessing.
                Xm = pre.transform(X_pred)
            preds, crep = cluster(Xm, ds.n_clusters, deadline, log=log,
                                  names=pre.feature_names_)
            report.update(crep)
    except Exception as e:
        log("modelling failed: " + "".join(traceback.format_exception(e)))
        report["error"] = f"{type(e).__name__}: {e}"
        preds = None
    if preds is not None:
        write_predictions(preds, ids, out_dir, ds.id_col)
    report["elapsed_seconds"] = round(time.time() - t_start, 2)
    write_json(report, out_dir, "run_report.json")
    log(f"done in {report['elapsed_seconds']}s")
    return report


def _write_fallback(ds, y, X_pred, ids, out_dir):
    try:
        if ds.task == "classification":
            v = y.mode(dropna=True).iloc[0]
            preds = np.full(len(X_pred), v, dtype=object)
        else:
            preds = np.full(len(X_pred), float(pd.to_numeric(y, errors="coerce").median()))
        write_predictions(preds, ids, out_dir, ds.id_col)
    except Exception:
        pass


def run_path(path=None, out_dir="output", time_limit=None, verbose=True, **kw) -> dict:
    ds = load_dataset(path, **kw)
    return run(ds, out_dir, time_limit=time_limit, verbose=verbose)
