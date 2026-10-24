"""Column type inference and the structured data profile."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from . import config

_NUM_CLEAN = re.compile(r"[,$€£%\s]")


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Parse strings like '1,234', '$5', '12%' to numbers; failures become NaN."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    st = s.astype("string").str.replace(_NUM_CLEAN, "", regex=True)
    return pd.to_numeric(st, errors="coerce").astype(float)


def _try_datetime(s: pd.Series) -> pd.Series | None:
    nonnull = s.dropna()
    if nonnull.empty:
        return None
    sample = nonnull.astype(str).head(200)
    # Require date-like separators so plain codes are not mistaken for dates.
    if sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}", regex=True).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if parsed.notna().sum() >= 0.9 * nonnull.shape[0]:
        return parsed
    return None


def infer_column_types(df: pd.DataFrame, hints: dict | None = None) -> dict:
    """Return {column: 'numeric' | 'categorical' | 'datetime' | 'text'}."""
    hints = hints or {}
    out = {}
    n = len(df)
    for c in df.columns:
        s = df[c]
        h = str(hints.get(c, "")).lower()
        if h in config.CATEGORICAL_TYPES:
            out[c] = "categorical"
            continue
        if h in config.NUMERIC_TYPES:
            out[c] = "numeric"
            continue
        if h in config.DATETIME_TYPES:
            out[c] = "datetime"
            continue
        if pd.api.types.is_bool_dtype(s):
            out[c] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            out[c] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = "numeric"
        else:
            nonnull = s.dropna()
            if nonnull.empty:
                out[c] = "categorical"
                continue
            num = to_numeric_clean(nonnull)
            if num.notna().mean() >= 0.95:
                out[c] = "numeric"
            elif _try_datetime(s) is not None:
                out[c] = "datetime"
            else:
                nun = nonnull.nunique()
                avg_len = nonnull.astype(str).str.len().mean()
                if nun > 0.5 * max(n, 1) and avg_len > 30:
                    out[c] = "text"
                else:
                    out[c] = "categorical"
    return out


def find_duplicate_columns(df: pd.DataFrame) -> dict:
    """{duplicate_col: original_col} for columns equal to an earlier one.
    Low-cardinality columns also match when equal up to a relabelling of
    values (e.g. 'a','b' vs 1,2)."""
    n = len(df)
    seen: dict = {}
    dups = {}
    for c in df.columns:
        s = df[c]
        try:
            nun = s.nunique(dropna=True)
            if pd.api.types.is_numeric_dtype(s):
                v = s.to_numpy(dtype=float)
                key = ("num", hash(np.round(v, 9).tobytes()), hash(np.isnan(v).tobytes()))
            else:
                key = ("obj", hash(tuple(s.astype("string").fillna("__nan__"))))
            if nun <= 50 and nun < n:
                codes, _ = pd.factorize(s.astype("string").fillna("__nan__"))
                key2 = ("rel", hash(codes.tobytes()))
            else:
                key2 = None
        except (TypeError, ValueError):
            continue
        orig = seen.get(key) or (seen.get(key2) if key2 else None)
        if orig is not None:
            dups[c] = orig
            continue
        seen[key] = c
        if key2:
            seen[key2] = c
    return dups


_HEX_ID = re.compile(r"^[0-9a-fA-F-]{16,}$")


def is_id_like(s: pd.Series, name: str, ctype: str, position: int = -1) -> bool:
    """Conservative identifier detection. A declared id column is handled by
    the caller; here a column is an id only with two pieces of evidence:
    unique values AND (an id-style name, or a 0/1-based consecutive integer
    sequence in the first column, or uuid/hash-like strings)."""
    n = len(s)
    if n < 20 or s.isna().any():
        return False
    nun = s.nunique()
    if nun != n:
        return False
    lname = name.lower().strip()
    named = lname in {"id", "index", "idx", "row_id", "rowid", "row", "uid", "uuid",
                      "key", "guid", "record_id", "customer_id", "user_id"} or (
        (lname.endswith("_id") or lname.endswith("id")) and len(lname) <= 24)
    if ctype == "numeric":
        if named:
            return True
        v = to_numeric_clean(s).to_numpy()
        if not np.all(np.isfinite(v)) or not np.all(np.mod(v, 1) == 0):
            return False
        sv = np.sort(v)
        consecutive = sv.size > 1 and np.all(np.diff(sv) == 1) and sv[0] in (0, 1)
        return bool(consecutive and position == 0)
    if ctype in ("categorical", "text"):
        if named:
            return True
        sample = s.astype(str).head(200)
        return bool(sample.str.match(_HEX_ID).mean() > 0.95)
    return False


def _num_stats(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {}
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    out_frac = float(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).mean()) if iqr > 0 else 0.0
    return {
        "mean": float(x.mean()), "std": float(x.std()) if len(x) > 1 else 0.0,
        "min": float(x.min()), "p25": float(q1), "median": float(x.median()),
        "p75": float(q3), "max": float(x.max()),
        "skewness": float(x.skew()) if len(x) > 2 else 0.0,
        "outlier_fraction": out_frac,
    }


def build_profile(ds, col_types: dict, drop_info: dict) -> dict:
    """Structured description of the dataset. `drop_info` comes from the
    preprocessor (constant / duplicate / id columns)."""
    train = ds.train
    feats = [c for c in train.columns if c != ds.target]
    X = train[feats]
    n = len(train)

    columns = []
    for c in feats:
        s = X[c]
        t = col_types.get(c, "unknown")
        info = {
            "name": c,
            "dtype": str(s.dtype),
            "inferred_type": t,
            "n_missing": int(s.isna().sum()),
            "missing_fraction": float(s.isna().mean()) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
            "is_constant": c in drop_info.get("constant", []),
            "duplicate_of": drop_info.get("duplicate", {}).get(c),
            "is_id_like": c in drop_info.get("id_like", []),
            "dropped": c in drop_info.get("dropped", []),
        }
        if t == "numeric":
            info["stats"] = _num_stats(to_numeric_clean(s))
        elif t in ("categorical", "text"):
            vc = s.astype("string").value_counts(dropna=True)
            info["top_values"] = {str(k): int(v) for k, v in vc.head(5).items()}
            info["high_cardinality"] = bool(vc.shape[0] > 30)
        columns.append(info)

    num_cols = [c for c in feats if col_types.get(c) == "numeric"]
    cat_cols = [c for c in feats if col_types.get(c) in ("categorical", "text")]
    dt_cols = [c for c in feats if col_types.get(c) == "datetime"]

    summary = {
        "n_rows": int(n),
        "n_rows_train": int(n),
        "n_rows_test": int(len(ds.test)) if ds.test is not None else 0,
        "n_columns": int(train.shape[1]),
        "n_features": int(len(feats)),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_datetime": len(dt_cols),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "datetime_columns": dt_cols,
        "missing_cells": int(X.isna().sum().sum()),
        "missing_fraction": float(X.isna().mean().mean()) if feats else 0.0,
        "columns_with_missing": [c for c in feats if X[c].isna().any()],
        "constant_columns": list(drop_info.get("constant", [])),
        "duplicate_columns": list(drop_info.get("duplicate", {}).keys()),
        "duplicate_column_pairs": drop_info.get("duplicate", {}),
        "id_like_columns": list(drop_info.get("id_like", [])),
        "high_cardinality_columns": [c["name"] for c in columns if c.get("high_cardinality")],
        "skewed_columns": [c["name"] for c in columns
                           if abs(c.get("stats", {}).get("skewness", 0)) > 1],
        "duplicate_rows": int(X.duplicated().sum()) if feats else 0,
    }
    # Scale spread: ratio between the largest and smallest numeric std.
    stds = [c["stats"]["std"] for c in columns if c.get("stats", {}).get("std", 0) > 0]
    summary["scale_ratio"] = float(max(stds) / min(stds)) if len(stds) > 1 else 1.0
    summary["mixed_scales"] = bool(summary["scale_ratio"] > 100)

    # Strongly correlated numeric pairs (|r| > 0.95), capped for size.
    corr_pairs = []
    if 1 < len(num_cols) <= 300:
        sub = train[num_cols].apply(to_numeric_clean)
        if n > 5000:
            sub = sub.sample(5000, random_state=0)
        cm = sub.corr().abs()
        iu = np.triu_indices_from(cm, k=1)
        for i, j in zip(*iu):
            v = cm.values[i, j]
            if np.isfinite(v) and v > 0.95:
                corr_pairs.append([num_cols[i], num_cols[j], round(float(v), 4)])
    summary["highly_correlated_pairs"] = corr_pairs[:50]

    profile = {
        "task": ds.task,
        "target": ds.target,
        "n_clusters": ds.n_clusters,
        "summary": summary,
        "columns": columns,
    }

    if ds.target is not None and ds.task != "clustering":
        y = train[ds.target]
        tgt = {"name": ds.target, "n_missing": int(y.isna().sum())}
        if ds.task == "classification":
            vc = y.value_counts(dropna=True)
            tgt["n_classes"] = int(vc.shape[0])
            tgt["class_counts"] = {str(k): int(v) for k, v in vc.items()}
            tgt["class_fractions"] = {str(k): float(v / vc.sum()) for k, v in vc.items()}
            ratio = float(vc.max() / vc.min()) if vc.min() > 0 else float("inf")
            tgt["imbalance_ratio"] = ratio
            tgt["is_imbalanced"] = bool(ratio > 3 or vc.min() / vc.sum() < 0.1)
            tgt["minority_class"] = str(vc.idxmin())
        else:
            yn = to_numeric_clean(y)
            tgt.update(_num_stats(yn))
            tgt["log_transform_recommended"] = bool(
                yn.min() >= 0 and abs(yn.skew()) > 1)
        profile["target_info"] = tgt
        relevance = feature_relevance(train, feats, ds.target, ds.task, col_types)
        if relevance:
            profile["feature_relevance"] = relevance
            summary["likely_irrelevant_features"] = [
                k for k, v in relevance.items() if v < 0.005]
    return profile


def feature_relevance(train, feats, target, task, col_types) -> dict:
    """Mutual information of each feature with the target (on a sample)."""
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        use = [c for c in feats if col_types.get(c) in ("numeric", "categorical")]
        if not use or len(use) > 500:
            return {}
        df = train[use + [target]].dropna(subset=[target])
        if len(df) > 3000:
            df = df.sample(3000, random_state=0)
        Xm = np.empty((len(df), len(use)))
        disc = []
        for i, c in enumerate(use):
            if col_types[c] == "numeric":
                v = to_numeric_clean(df[c])
                Xm[:, i] = v.fillna(v.median() if v.notna().any() else 0).to_numpy()
                disc.append(False)
            else:
                Xm[:, i] = pd.factorize(df[c].astype("string"))[0]
                disc.append(True)
        y = df[target]
        if task == "classification":
            mi = mutual_info_classif(Xm, pd.factorize(y)[0], discrete_features=disc,
                                     random_state=0)
        else:
            mi = mutual_info_regression(Xm, to_numeric_clean(y).to_numpy(),
                                        discrete_features=disc, random_state=0)
        return {c: round(float(v), 5) for c, v in zip(use, mi)}
    except Exception:
        return {}
