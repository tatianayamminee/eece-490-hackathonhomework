"""Time-budgeted model selection for classification and regression.

Each candidate is a (preprocessing mode, estimator) pair scored with K-fold
cross-validation on out-of-fold predictions. Candidates run cheapest/strongest
first and are skipped when the remaining budget cannot cover them. The best
single model is compared with a greedy ensemble of the candidates' out-of-fold
predictions; the winner is refit on all training data.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from . import config
from .preprocess import Preprocessor

warnings.filterwarnings("ignore")
N_JOBS = max(1, os.cpu_count() or 1)


def _lgbm(task, **kw):
    import lightgbm as lgb
    base = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=N_JOBS,
                random_state=config.RANDOM_STATE, verbose=-1)
    base.update(kw)
    return lgb.LGBMClassifier(**base) if task == "classification" else lgb.LGBMRegressor(**base)


def candidate_models(task: str, n: int, p: int, n_classes: int = 0, imbalanced=False):
    """Ordered list of (name, preprocessing mode, estimator, log_target)."""
    small = n <= 3000
    c = []
    if task == "classification":
        c.append(("lgbm", "tree", _lgbm(task), False))
        c.append(("logreg", "linear", LogisticRegression(C=1.0, max_iter=2000), False))
        if imbalanced:
            c.append(("lgbm_balanced", "tree", _lgbm(task, class_weight="balanced"), False))
            c.append(("logreg_balanced", "linear",
                      LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"), False))
        c.append(("extra_trees", "tree", ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=1, max_features="sqrt", n_jobs=N_JOBS,
            random_state=config.RANDOM_STATE,
            class_weight="balanced_subsample" if imbalanced else None), False))
        c.append(("random_forest", "tree", RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=N_JOBS,
            random_state=config.RANDOM_STATE,
            class_weight="balanced_subsample" if imbalanced else None), False))
        c.append(("lgbm_slow", "tree", _lgbm(task, n_estimators=1000, learning_rate=0.02,
                                              num_leaves=15), False))
        if small:
            c.append(("svm_rbf", "linear", SVC(C=3.0, probability=True,
                                               random_state=config.RANDOM_STATE,
                                               class_weight="balanced" if imbalanced else None),
                      False))
            c.append(("knn", "linear", KNeighborsClassifier(
                n_neighbors=15, weights="distance", n_jobs=N_JOBS), False))
    else:
        c.append(("lgbm", "tree", _lgbm(task), False))
        c.append(("ridge", "linear", RidgeCV(alphas=np.logspace(-3, 3, 13)), False))
        c.append(("lgbm_logy", "tree", _lgbm(task), True))
        c.append(("extra_trees", "tree", ExtraTreesRegressor(
            n_estimators=300, max_features=0.7, n_jobs=N_JOBS,
            random_state=config.RANDOM_STATE), False))
        c.append(("random_forest", "tree", RandomForestRegressor(
            n_estimators=300, max_features=0.5, n_jobs=N_JOBS,
            random_state=config.RANDOM_STATE), False))
        c.append(("lgbm_slow", "tree", _lgbm(task, n_estimators=1000, learning_rate=0.02,
                                              num_leaves=15), False))
        c.append(("ridge_logy", "linear", RidgeCV(alphas=np.logspace(-3, 3, 13)), True))
        if small:
            c.append(("svr_rbf", "linear", SVR(C=3.0), False))
            c.append(("knn", "linear", KNeighborsRegressor(
                n_neighbors=10, weights="distance", n_jobs=N_JOBS), False))
    return c


# ----------------------------------------------------------------- metrics
def score(task: str, y_true, pred, proba=None, metric: str | None = None) -> float:
    """Higher is better for every metric (errors are negated)."""
    m = metric or (config.CLASSIFICATION_METRIC if task == "classification"
                   else config.REGRESSION_METRIC)
    if m == "accuracy":
        return accuracy_score(y_true, pred)
    if m == "balanced_accuracy":
        return balanced_accuracy_score(y_true, pred)
    if m == "f1_macro":
        return f1_score(y_true, pred, average="macro")
    if m == "f1_weighted":
        return f1_score(y_true, pred, average="weighted")
    if m == "f1":
        return f1_score(y_true, pred)
    if m == "roc_auc":
        return roc_auc_score(y_true, proba[:, 1]) if proba.shape[1] == 2 else \
            roc_auc_score(y_true, proba, multi_class="ovr")
    if m == "log_loss":
        return -log_loss(y_true, np.clip(proba, 1e-15, 1))
    if m == "rmse":
        return -np.sqrt(mean_squared_error(y_true, pred))
    if m == "mae":
        return -mean_absolute_error(y_true, pred)
    if m == "r2":
        return r2_score(y_true, pred)
    if m == "rmsle":
        return -np.sqrt(mean_squared_error(np.log1p(np.clip(y_true, 0, None)),
                                           np.log1p(np.clip(pred, 0, None))))
    raise ValueError(m)


# ---------------------------------------------------------------- fitting
class FittedModel:
    """Preprocessor + estimator (+ optional log target), fit on raw frames."""

    def __init__(self, mode, est, log_target, col_types, drop_info, hints):
        self.pre = Preprocessor(mode=mode, hints=hints, col_types=col_types,
                                drop_info=drop_info)
        self.est = clone(est)
        self.log_target = log_target

    def fit(self, X, y):
        Xm = self.pre.fit_transform(X)
        yt = np.log1p(y) if self.log_target else y
        self.est.fit(Xm, yt)
        return self

    def predict_raw(self, X, task):
        Xm = self.pre.transform(X)
        if task == "classification":
            return self.est.predict_proba(Xm)
        p = self.est.predict(Xm)
        return np.expm1(p) if self.log_target else p


class SupervisedAutoML:
    def __init__(self, task, deadline, col_types, drop_info, hints=None, log=print):
        self.task = task
        self.deadline = deadline
        self.col_types = col_types
        self.drop_info = drop_info
        self.hints = hints or {}
        self.log = log
        self.report = {"candidates": []}

    def remaining(self):
        return self.deadline - time.time()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        task = self.task
        mask = y.notna()
        X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
        if task == "classification":
            self.classes_, y_enc = np.unique(y.astype(str), return_inverse=True)
            self._orig_labels = {str(v): v for v in y.unique()}
            yv = y_enc
            counts = np.bincount(yv)
            imbalanced = counts.min() / counts.sum() < 0.15 or counts.max() / counts.min() > 3
        else:
            from .profile import to_numeric_clean
            yv = to_numeric_clean(y).to_numpy()
            ok = np.isfinite(yv)
            X, yv = X.loc[ok].reset_index(drop=True), yv[ok]
            imbalanced = False
        n, p = X.shape
        self.n_train = n

        # Subsample for selection on large data; final refit uses everything.
        sel_idx = np.arange(n)
        if n > 20000:
            rs = np.random.RandomState(config.RANDOM_STATE)
            sel_idx = np.sort(rs.choice(n, 20000, replace=False))
        Xs, ys = X.iloc[sel_idx].reset_index(drop=True), yv[sel_idx]

        n_folds = 5 if len(Xs) <= 5000 else 3
        if task == "classification":
            min_class = np.bincount(ys).min()
            n_folds = max(2, min(n_folds, int(min_class)))
            cv = StratifiedKFold(n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        else:
            cv = KFold(n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        folds = list(cv.split(Xs, ys))
        n_classes = len(self.classes_) if task == "classification" else 0

        can_log = task == "regression" and ys.min() >= 0
        cands = [c for c in candidate_models(task, len(Xs), p, n_classes, imbalanced)
                 if not (c[3] and not can_log)]

        oofs, results = {}, []
        scale_up = n / max(len(Xs), 1)
        refit_reserve = 0.0  # projected time to refit the best model on all rows
        for name, mode, est, logy in cands:
            budget = self.remaining() - refit_reserve
            if budget < 3:
                break
            if logy and task == "regression" and abs(pd.Series(ys).skew()) < 0.5:
                continue
            t0 = time.time()
            oof = np.zeros((len(Xs), n_classes)) if task == "classification" \
                else np.zeros(len(Xs))
            ok = True
            for i, (tr, va) in enumerate(folds):
                try:
                    fm = FittedModel(mode, est, logy, self.col_types, self.drop_info, self.hints)
                    fm.fit(Xs.iloc[tr], ys[tr])
                    pr = fm.predict_raw(Xs.iloc[va], task)
                    if task == "classification":
                        full = np.zeros((len(va), n_classes))
                        full[:, fm.est.classes_] = pr
                        pr = full
                    oof[va] = pr
                except Exception as e:  # a failing candidate never kills the run
                    self.log(f"  {name}: failed ({type(e).__name__}: {e})")
                    ok = False
                    break
                el = time.time() - t0
                # Projected cost of the remaining folds plus a final refit.
                proj = el / (i + 1) * (len(folds) - i - 1 + 1.5 * scale_up)
                if i == 0 and proj > self.remaining() - 2 - refit_reserve:
                    self.log(f"  {name}: skipped (needs ~{proj:.1f}s)")
                    ok = False
                    break
            if not ok:
                continue
            fit_time = (time.time() - t0) / len(folds)
            s = self._score_oof(ys, oof)
            oofs[name] = oof
            results.append({"name": name, "mode": mode, "log_target": logy,
                            "cv_score": float(s), "fit_seconds": round(fit_time, 3),
                            "est": est})
            self.log(f"  {name:16s} cv={s:.4f}  ({fit_time:.2f}s/fold)")
            best_so_far = max(results, key=lambda r: r["cv_score"])
            refit_reserve = best_so_far["fit_seconds"] * 1.5 * scale_up

        if not results:
            raise RuntimeError("no candidate finished in time")
        results.sort(key=lambda r: -r["cv_score"])
        self.report["candidates"] = [{k: v for k, v in r.items() if k != "est"}
                                     for r in results]
        self.report["metric"] = (config.CLASSIFICATION_METRIC if task == "classification"
                                 else config.REGRESSION_METRIC)
        self.report["cv_folds"] = n_folds

        weights = self._greedy_ensemble(ys, oofs, results)
        ens_oof = sum(w * oofs[k] for k, w in weights.items())
        ens_score = self._score_oof(ys, ens_oof)
        best = results[0]
        if ens_score > best["cv_score"] + 1e-6 and len(weights) > 1:
            chosen = weights
            self.report["ensemble"] = {k: round(v, 3) for k, v in weights.items()}
            self.report["cv_score"] = float(ens_score)
            final_oof = ens_oof
        else:
            chosen = {best["name"]: 1.0}
            self.report["cv_score"] = best["cv_score"]
            final_oof = oofs[best["name"]]
        self.report["chosen"] = chosen

        # Class-weight tuning for macro-F1 / balanced accuracy on OOF probabilities.
        self.class_scale = None
        if task == "classification" and self.report["metric"] in (
                "f1_macro", "balanced_accuracy", "f1"):
            self.class_scale = self._tune_class_scale(ys, final_oof)
            if self.class_scale is not None:
                self.report["class_scale"] = self.class_scale.round(3).tolist()
                self.report["cv_score_after_scaling"] = float(self._score_oof(
                    ys, final_oof, self.class_scale))

        # Refit chosen models on all rows (drop the slowest if time is short).
        by_name = {r["name"]: r for r in results}
        self.models = []
        for name, w in sorted(chosen.items(), key=lambda kv: -kv[1]):
            r = by_name[name]
            need = r["fit_seconds"] * 1.5 * (n / max(len(Xs), 1))
            if self.models and need > self.remaining() - 1:
                self.log(f"  refit {name} skipped for time")
                continue
            fm = FittedModel(r["mode"], r["est"], r["log_target"], self.col_types,
                             self.drop_info, self.hints).fit(X, yv)
            self.models.append((fm, w))
        tot = sum(w for _, w in self.models)
        self.models = [(m, w / tot) for m, w in self.models]
        return self

    # ------------------------------------------------------------ helpers
    def _score_oof(self, y, oof, scale=None):
        if self.task == "classification":
            p = oof * scale if scale is not None else oof
            return score("classification", y, p.argmax(1), proba=oof)
        return score("regression", y, oof)

    def _greedy_ensemble(self, y, oofs, results, rounds=15):
        """Caruana-style forward selection with replacement over the top models."""
        names = [r["name"] for r in results[:6]]
        if len(names) < 2:
            return {names[0]: 1.0}
        counts = {k: 0 for k in names}
        cur, k = None, 0
        best_s = -np.inf
        for _ in range(rounds):
            cand_best, cand_s = None, -np.inf
            for nm in names:
                trial = oofs[nm] if cur is None else (cur * k + oofs[nm]) / (k + 1)
                s = self._score_oof(y, trial)
                if s > cand_s:
                    cand_best, cand_s = nm, s
            if cand_s <= best_s + 1e-9 and cur is not None:
                break
            cur = oofs[cand_best] if cur is None else (cur * k + oofs[cand_best]) / (k + 1)
            k += 1
            counts[cand_best] += 1
            best_s = cand_s
        return {nm: c / k for nm, c in counts.items() if c > 0}

    def _tune_class_scale(self, y, proba):
        K = proba.shape[1]
        scale = np.ones(K)
        base = self._score_oof(y, proba)
        best = base
        grid = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
        for _ in range(2):
            for k in range(K):
                for g in grid:
                    s2 = scale.copy()
                    s2[k] = g
                    v = self._score_oof(y, proba, s2)
                    if v > best + 1e-6:
                        best, scale = v, s2
        # Require a meaningful gain to avoid overfitting the OOF set.
        return scale if best > base + 0.002 else None

    def predict(self, X: pd.DataFrame):
        out = None
        for fm, w in self.models:
            p = fm.predict_raw(X, self.task)
            if self.task == "classification":
                full = np.zeros((len(X), len(self.classes_)))
                full[:, fm.est.classes_] = p
                p = full
            out = p * w if out is None else out + p * w
        if self.task == "classification":
            if self.class_scale is not None:
                out = out * self.class_scale
            labels = self.classes_[out.argmax(1)]
            return np.array([self._orig_labels.get(l, l) for l in labels], dtype=object)
        return out
