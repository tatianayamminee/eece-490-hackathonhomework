"""Turn a messy DataFrame into a clean numeric matrix.

Two flavours, picked per model:
  * mode="tree":   ordinal-coded categoricals, no scaling (for boosting / forests)
  * mode="linear": one-hot categoricals, skew-corrected and standardised numerics
                   (for linear models, kNN, SVM and clustering)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profile import (_try_datetime, find_duplicate_columns, infer_column_types,
                      is_id_like, to_numeric_clean)

ONEHOT_MAX_LEVELS = 20
RARE_MIN_COUNT = 5


def analyse_columns(X: pd.DataFrame, hints: dict | None = None):
    """Column types plus which columns to drop and why (fit on training data)."""
    types = infer_column_types(X, hints)
    constant, id_like = [], []
    for c in X.columns:
        nun = X[c].nunique(dropna=True)
        if nun <= 1 and not (nun == 1 and X[c].isna().any()):
            constant.append(c)
        elif is_id_like(X[c], c, types[c]) and not (hints or {}).get(c):
            id_like.append(c)
    rest = X[[c for c in X.columns if c not in constant and c not in id_like]]
    duplicate = find_duplicate_columns(rest)
    dropped = constant + id_like + list(duplicate)
    return types, {"constant": constant, "id_like": id_like,
                   "duplicate": duplicate, "dropped": dropped}


class Preprocessor:
    def __init__(self, mode: str = "tree", hints: dict | None = None,
                 col_types: dict | None = None, drop_info: dict | None = None,
                 max_onehot: int = ONEHOT_MAX_LEVELS):
        self.mode = mode
        self.hints = hints or {}
        self.col_types = col_types
        self.drop_info = drop_info
        self.max_onehot = max_onehot

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None):
        if self.col_types is None or self.drop_info is None:
            self.col_types, self.drop_info = analyse_columns(X, self.hints)
        keep = [c for c in X.columns if c not in set(self.drop_info["dropped"])]
        self.num_cols = [c for c in keep if self.col_types[c] == "numeric"]
        self.cat_cols = [c for c in keep if self.col_types[c] in ("categorical", "text")]
        self.dt_cols = [c for c in keep if self.col_types[c] == "datetime"]

        N = self._numeric_frame(X)
        self.missing_ind = [c for c in N.columns if N[c].isna().mean() > 0.0]
        # Sign-preserving log for heavily skewed columns (linear mode only).
        self.log_cols = []
        if self.mode == "linear":
            for c in N.columns:
                v = N[c].dropna()
                if len(v) > 10 and abs(v.skew()) > 2 and v.nunique() > 10:
                    self.log_cols.append(c)
        Nf = self._num_transform(N)
        self.medians_t = Nf.median().fillna(0.0)
        lo, hi = Nf.quantile(0.001), Nf.quantile(0.999)
        self.clip_lo, self.clip_hi = lo, hi

        # Categoricals: keep levels seen at least RARE_MIN_COUNT times.
        self.cat_levels = {}
        self.cat_freq = {}
        for c in self.cat_cols:
            s = self._cat_series(X[c])
            vc = s.value_counts()
            self.cat_freq[c] = (vc / max(len(s), 1)).to_dict()
            levels = vc[vc >= min(RARE_MIN_COUNT, max(2, len(s) // 200))].index.tolist()
            self.cat_levels[c] = levels

        if self.mode == "linear":
            M = self._assemble(X, scale=False)
            self.mean_ = M.mean(axis=0)
            sd = M.std(axis=0)
            self.std_ = np.where(sd > 1e-12, sd, 1.0)
        self.feature_names_ = list(self._assemble(X.head(1), scale=False, names_only=True))
        return self

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _cat_series(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip()
        return s.fillna("__nan__")

    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = {c: to_numeric_clean(X[c]) for c in self.num_cols}
        for c in self.dt_cols:
            d = X[c] if pd.api.types.is_datetime64_any_dtype(X[c]) else _try_datetime(X[c])
            if d is None:
                d = pd.to_datetime(X[c], errors="coerce", format="mixed")
            d = pd.to_datetime(d, errors="coerce")
            parts[f"{c}__year"] = d.dt.year.astype(float)
            parts[f"{c}__month"] = d.dt.month.astype(float)
            parts[f"{c}__day"] = d.dt.day.astype(float)
            parts[f"{c}__dow"] = d.dt.dayofweek.astype(float)
            parts[f"{c}__hour"] = d.dt.hour.astype(float)
            parts[f"{c}__ord"] = (d - pd.Timestamp("1970-01-01")).dt.days.astype(float)
        N = pd.DataFrame(parts, index=X.index)
        return N.replace([np.inf, -np.inf], np.nan)

    def _num_transform(self, N: pd.DataFrame) -> pd.DataFrame:
        N = N.copy()
        for c in self.log_cols:
            N[c] = np.sign(N[c]) * np.log1p(np.abs(N[c]))
        return N

    def _assemble(self, X: pd.DataFrame, scale=True, names_only=False):
        N = self._num_transform(self._numeric_frame(X))
        names, blocks = [], []
        if len(N.columns):
            ind = N[self.missing_ind].isna().to_numpy(dtype=float) if self.missing_ind else None
            if hasattr(self, "clip_lo"):
                N = N.clip(self.clip_lo, self.clip_hi, axis=1)
            Nv = N.fillna(self.medians_t)
            blocks.append(Nv.to_numpy(dtype=float))
            names += list(N.columns)
            if ind is not None:
                blocks.append(ind)
                names += [f"{c}__missing" for c in self.missing_ind]
        for c in self.cat_cols:
            s = self._cat_series(X[c])
            levels = self.cat_levels.get(c, [])
            freq = s.map(self.cat_freq.get(c, {})).astype(float).fillna(0.0).to_numpy()
            if self.mode == "linear" and 0 < len(levels) <= self.max_onehot:
                lv = levels[1:] if len(levels) == 2 else levels
                oh = np.stack([(s == l).to_numpy(dtype=float) for l in lv], axis=1)
                blocks.append(oh)
                names += [f"{c}=={l}" for l in lv]
                if len(levels) > 2:
                    blocks.append(freq[:, None])
                    names.append(f"{c}__freq")
            else:
                code = pd.Categorical(s, categories=levels).codes.astype(float)
                if self.mode == "tree":
                    blocks.append(code[:, None])
                    names.append(f"{c}__code")
                blocks.append(freq[:, None])
                names.append(f"{c}__freq")
        if names_only:
            return names
        M = np.hstack(blocks) if blocks else np.zeros((len(X), 1))
        if scale and self.mode == "linear":
            M = (M - self.mean_) / self.std_
        return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._assemble(X, scale=True)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
