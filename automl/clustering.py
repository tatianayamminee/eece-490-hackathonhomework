"""Clustering with a fixed number of clusters.

Several candidate partitions are built from different views of the data
(all features, PCA, an automatically selected feature subset, Gaussian
mixtures, Ward linkage). Irrelevant noise columns are the main way clustering
goes wrong, so the feature-subset view iteratively drops columns that do not
separate the current clusters. The final partition is chosen by a mix of
internal quality (silhouette on the selected features) and agreement with the
other candidates (partitions many methods agree on are usually the real ones).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from . import config

warnings.filterwarnings("ignore")
RS = config.RANDOM_STATE


def _kmeans(X, k, n_init=10):
    return KMeans(k, n_init=n_init, random_state=RS).fit_predict(X)


def _f_stats(X, labels):
    """Per-feature ratio of between-cluster to within-cluster variance."""
    overall = X.mean(0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in np.unique(labels):
        m = labels == c
        if m.sum() == 0:
            continue
        mu = X[m].mean(0)
        between += m.sum() * (mu - overall) ** 2
        within += ((X[m] - mu) ** 2).sum(0)
    return between / np.maximum(within, 1e-12)


def select_features(X, k, iters=4):
    """Drop features that do not separate the clusters (sparse k-means style)."""
    keep = np.arange(X.shape[1])
    labels = _kmeans(X, k)
    for _ in range(iters):
        f = np.minimum(_f_stats(X[:, keep], labels), 1e6)
        if f.max() <= 0:
            break
        # Noise features separate nothing (F near 0); real ones are well above.
        thr = max(0.1, 0.02 * f.max())
        new_keep = keep[f >= thr]
        if len(new_keep) == 0:
            break
        if len(new_keep) == len(keep):
            break
        keep = new_keep
        labels = _kmeans(X[:, keep], k)
    return keep, labels


def _sil(X, labels, sample=3000):
    if len(np.unique(labels)) < 2:
        return -1.0
    try:
        return float(silhouette_score(X, labels, sample_size=min(sample, len(X)),
                                      random_state=RS))
    except Exception:
        return -1.0


def choose_k(X, kmax=10):
    best_k, best_s = 2, -np.inf
    for k in range(2, min(kmax, len(X) - 1) + 1):
        s = _sil(X, _kmeans(X, k, n_init=4))
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def weight_columns(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Standardised one-hot columns dominate distances; shrink each dummy group
    to the weight of one numeric column and drop the redundant frequency codes."""
    keep, w = [], []
    groups: dict = {}
    for i, nm in enumerate(names):
        if "==" in nm:
            groups.setdefault(nm.split("==")[0], []).append(i)
    for i, nm in enumerate(names):
        if nm.endswith("__freq") and nm[:-6] in groups:
            continue
        keep.append(i)
        g = nm.split("==")[0] if "==" in nm else None
        w.append(1.0 / np.sqrt(len(groups[g])) if g else 1.0)
    return X[:, keep] * np.asarray(w)


def cluster(X: np.ndarray, k: int | None, deadline: float, log=print, names=None):
    report = {}
    if names is not None and len(names) == X.shape[1]:
        X = weight_columns(X, names)
    n, p = X.shape
    if k is None:
        k = choose_k(X)
        report["k_chosen_automatically"] = k
    k = int(max(1, min(k, n)))
    if k == 1:
        return np.zeros(n, dtype=int), {"k": 1}

    cands = {}
    t0 = time.time()

    def left():
        return deadline - time.time()

    cands["kmeans_all"] = _kmeans(X, k)
    keep, lab_sel = select_features(X, k)
    Xsel = X[:, keep]
    cands["kmeans_selected"] = lab_sel
    report["selected_features"] = int(len(keep))

    if p > 2 and left() > 5:
        pca = PCA(n_components=0.9, svd_solver="full", random_state=RS).fit(X)
        Xp = pca.transform(X)
        cands["kmeans_pca"] = _kmeans(Xp, k)
    if left() > 5 and n <= 50000:
        for name, Z in (("gmm_selected", Xsel),):
            try:
                cov = "full" if Z.shape[1] <= 20 else "diag"
                g = GaussianMixture(k, covariance_type=cov, n_init=3, random_state=RS,
                                    reg_covar=1e-4).fit(Z)
                cands[name] = g.predict(Z)
            except Exception:
                pass
    if left() > 5 and n <= 8000:
        cands["ward_selected"] = AgglomerativeClustering(k, linkage="ward").fit_predict(Xsel)
    if left() > 5 and n <= 8000:
        cands["ward_all"] = AgglomerativeClustering(k, linkage="ward").fit_predict(X)

    names = list(cands)
    sil = {nm: _sil(Xsel, cands[nm]) for nm in names}
    agree = {}
    for a in names:
        others = [adjusted_rand_score(cands[a], cands[b]) for b in names if b != a]
        agree[a] = float(np.mean(others)) if others else 0.0
    # Rank-average of silhouette (on selected features) and agreement.
    def rank(d):
        order = sorted(d, key=lambda x: d[x])
        return {nm: i / max(len(order) - 1, 1) for i, nm in enumerate(order)}
    rs, ra = rank(sil), rank(agree)
    combo = {nm: 0.5 * rs[nm] + 0.5 * ra[nm] for nm in names}
    best = max(names, key=lambda nm: (combo[nm], sil[nm]))
    report.update({
        "k": k, "candidates": {nm: {"silhouette": round(sil[nm], 4),
                                    "agreement": round(agree[nm], 4)} for nm in names},
        "chosen": best, "seconds": round(time.time() - t0, 2),
    })
    log(f"  clustering candidates: {report['candidates']}  -> {best}")
    labels = cands[best]
    # Relabel 0..k-1 by cluster size for stable output.
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[inv], report
