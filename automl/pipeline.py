"""End-to-end run for one dataset: load -> profile -> model -> write outputs."""
from __future__ import annotations

import os
import sys
import threading
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


def run(ds: Dataset, out_dir: str, time_limit: float | None = None, verbose=True,
        hard_limit: bool = False) -> dict:
    """Process one dataset. With `hard_limit`, a watchdog thread terminates the
    process shortly before `time_limit` (the fallback predictions and the
    profile are already on disk by then), so the limit can never be exceeded."""
    t_start = time.time()
    log = _log if verbose else (lambda m: None)
    time_limit = time_limit or config.TIME_LIMIT_SECONDS
    deadline = t_start + time_limit - config.SAFETY_MARGIN_SECONDS
    watchdog = None
    if hard_limit:
        def _abort():
            write_json({"task": ds.task, "timed_out": True,
                        "elapsed_seconds": round(time.time() - t_start, 2)},
                       out_dir, "run_report.json")
            log("watchdog: time limit reached, keeping fallback outputs")
            os._exit(0)
        watchdog = threading.Timer(max(1.0, time_limit - config.WATCHDOG_MARGIN_SECONDS),
                                   _abort)
        watchdog.daemon = True
        watchdog.start()
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

    # 1. Fallback predictions (majority class / median / quick k-means) and the
    #    profile go to disk first, so a timeout or crash never leaves no output.
    y = ds.train[ds.target] if ds.task != "clustering" else None
    pre = None
    if ds.task == "clustering":
        pre = Preprocessor(mode="linear", hints=ds.feature_types,
                           col_types=col_types, drop_info=drop_info, max_onehot=50)
        Xm = pre.fit_transform(X)
        if ds.test is not None:
            # Label the test rows; train rows only define the preprocessing.
            Xm = pre.transform(X_pred)
    else:
        Xm = None
    _write_fallback(ds, y, X_pred, Xm, ids, out_dir)
    profile = build_profile(ds, col_types, drop_info)
    write_json(profile, out_dir, config.PROFILE_FILE)
    log(f"fallback predictions and profile written ({time.time() - t_start:.1f}s)")

    # 2. model
    report: dict = {"task": ds.task}
    try:
        if ds.task in ("classification", "regression"):
            aml = SupervisedAutoML(ds.task, deadline, col_types, drop_info,
                                   ds.feature_types, log=log)
            aml.fit(X, y)
            preds = aml.predict(X_pred)
            report.update(aml.report)
        else:
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
    if watchdog is not None:
        watchdog.cancel()
    log(f"done in {report['elapsed_seconds']}s")
    return report


def _write_fallback(ds, y, X_pred, Xm, ids, out_dir):
    try:
        if ds.task == "classification":
            v = y.mode(dropna=True).iloc[0]
            preds = np.full(len(X_pred), v, dtype=object)
        elif ds.task == "regression":
            preds = np.full(len(X_pred), float(pd.to_numeric(y, errors="coerce").median()))
        else:
            from sklearn.cluster import KMeans
            k = ds.n_clusters or 2
            if Xm is not None and len(Xm) >= k:
                preds = KMeans(k, n_init=1, max_iter=30, random_state=0).fit_predict(Xm)
            else:
                preds = np.arange(len(X_pred)) % k
        write_predictions(preds, ids, out_dir, ds.id_col)
    except Exception:
        pass


def run_path(path=None, out_dir="output", time_limit=None, verbose=True,
             hard_limit=False, **kw) -> dict:
    ds = load_dataset(path, **kw)
    return run(ds, out_dir, time_limit=time_limit, verbose=verbose, hard_limit=hard_limit)
