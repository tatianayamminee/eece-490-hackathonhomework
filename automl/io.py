"""Loading a dataset and writing outputs. Contract details come from config.py."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config

NA_TOKENS = ["", "?", "NA", "N/A", "n/a", "na", "NaN", "nan", "null", "NULL", "None",
             "none", "-", "--", "missing", "MISSING", "#N/A", "inf", "-inf"]


@dataclass
class Dataset:
    task: str
    train: pd.DataFrame
    test: pd.DataFrame | None
    target: str | None
    n_clusters: int | None = None
    id_col: str | None = None
    feature_types: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    source: str = ""


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    sep = "\t" if path.endswith((".tsv", ".tab")) else _sniff_sep(path)
    df = pd.read_csv(path, sep=sep, na_values=NA_TOKENS, keep_default_na=True,
                     skipinitialspace=True, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _sniff_sep(path: str) -> str:
    with open(path, "r", errors="replace") as fh:
        head = "".join(fh.readline() for _ in range(5))
    counts = {s: head.count(s) for s in [",", ";", "\t", "|"]}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _first(d: dict, keys: list[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def canonical_task(value) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    for name, aliases in config.TASK_ALIASES.items():
        if v in aliases or v.startswith(name[:5]):
            return name
    return None


def _find(directory: str, names: list[str]) -> str | None:
    for n in names:
        p = os.path.join(directory, n)
        if os.path.exists(p):
            return p
    return None


def _normalise_feature_types(raw) -> dict:
    """Accepts {col: type}, [{name, type}], or {type: [cols]}; returns {col: type}."""
    out = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, (list, tuple)):          # {"categorical": [...]}
                for c in v:
                    out[str(c)] = str(k).lower()
            elif isinstance(v, dict):                  # {"col": {"type": ...}}
                t = _first(v, ["type", "dtype", "kind"])
                if t is not None:
                    out[str(k)] = str(t).lower()
            else:
                out[str(k)] = str(v).lower()
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                n = _first(item, ["name", "column", "feature"])
                t = _first(item, ["type", "dtype", "kind"])
                if n is not None and t is not None:
                    out[str(n)] = str(t).lower()
    return out


def load_dataset(path: str | None = None, *, train: str | None = None,
                 test: str | None = None, meta: str | None = None,
                 task: str | None = None, target: str | None = None,
                 n_clusters: int | None = None, id_col: str | None = None) -> Dataset:
    """Load from a dataset directory and/or explicit file paths. CLI values win."""
    meta_d: dict = {}
    if path and os.path.isdir(path):
        train = train or _find(path, config.TRAIN_FILES)
        test = test or _find(path, config.TEST_FILES)
        meta = meta or _find(path, config.META_FILES)
    elif path and os.path.isfile(path):
        if path.endswith(".json") and train:
            meta = meta or path
        else:
            train = train or path
    if meta:
        with open(meta) as fh:
            meta_d = json.load(fh)
    if not train:
        raise FileNotFoundError(f"No training table found (looked in {path!r})")

    train_df = read_table(train)
    test_df = read_table(test) if test else None

    task = canonical_task(task) or canonical_task(_first(meta_d, config.META_KEYS["task"]))
    target = target or _first(meta_d, config.META_KEYS["target"])
    if isinstance(target, list):
        target = target[0] if target else None
    k = n_clusters or _first(meta_d, config.META_KEYS["n_clusters"])
    id_col = id_col or _first(meta_d, config.META_KEYS["id"])
    ftypes = _normalise_feature_types(_first(meta_d, config.META_KEYS["feature_types"]) or {})

    # Guess the target when not given: a column present in train but not in test,
    # else a conventionally named column.
    if target is None and task != "clustering":
        if test_df is not None:
            extra = [c for c in train_df.columns if c not in set(test_df.columns)]
            if len(extra) == 1:
                target = extra[0]
        if target is None:
            for c in train_df.columns:
                if c.lower() in {"target", "label", "y", "class", "outcome"}:
                    target = c
                    break
        if target is None and task in ("classification", "regression"):
            target = train_df.columns[-1]
    if target is not None and target not in train_df.columns:
        target = None

    if task is None:
        if k is not None and target is None:
            task = "clustering"
        elif target is not None:
            y = train_df[target]
            task = ("classification" if (y.dtype == object or y.nunique() <= 20)
                    else "regression")
        else:
            task = "clustering"
    if task == "clustering" and k is None:
        k = None  # chosen automatically later
    if id_col is not None and id_col not in train_df.columns and (
            test_df is None or id_col not in test_df.columns):
        id_col = None

    return Dataset(task=task, train=train_df, test=test_df, target=target,
                   n_clusters=int(k) if k is not None else None, id_col=id_col,
                   feature_types=ftypes, meta=meta_d, source=str(path or train))


def write_predictions(preds, ids, out_dir: str, id_name: str | None):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({config.PREDICTION_COLUMN: np.asarray(preds)})
    if config.WRITE_ID_COLUMN and ids is not None:
        df.insert(0, id_name or "id", np.asarray(ids))
    p = os.path.join(out_dir, config.PREDICTIONS_FILE)
    tmp = p + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)  # atomic: a timeout never leaves a half-written file
    return p


def write_json(obj, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    tmp = p + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    os.replace(tmp, p)
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
