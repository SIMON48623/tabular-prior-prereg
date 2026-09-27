"""Frozen definitions of the alternative dispersion measures and the medical-tag candidate rule
(Phase 0c, computed before registration; no model comparison is involved).

The executor imports this file unchanged. It must not re-implement these functions.

1. neff_cluster      denoised N_eff after merging strongly correlated features into clusters
2. neff_conditional  N_eff of the out-of-fold contributions of each original feature to the linear
                     predictor of the Phase 0b reference logistic regression (conditional dispersion)
3. medical_candidate keyword rule that flags datasets for the manual medical-subset review
"""
from __future__ import annotations

import re

import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import stratifier  # noqa: E402

CLUSTER_ABS_RHO = 0.7          # features with |Spearman rho| >= 0.7 (average linkage) share a cluster


def _neff_from_lifts(lifts) -> float:
    lifts = np.asarray(lifts, float)
    s2 = float((lifts ** 2).sum())
    return float(lifts.sum() ** 2 / s2) if s2 > 0 else float("nan")


# --------------------------------------------------------------------------------------------------
# 1. cluster-adjusted N_eff
# --------------------------------------------------------------------------------------------------

def feature_clusters(X_codes: np.ndarray, names: list[str]) -> dict[str, int]:
    """Cluster features by absolute Spearman correlation.

    X_codes : (n, p) array of the cleaned features BEFORE imputation, with categorical columns as
              the integer codes used by Phase 0b (missing values as NaN). No label is used.
    Pairwise |rho| uses pairwise-complete rows; a pair with fewer than 20 complete rows, or a constant
    column, gets rho = 0. Average-linkage hierarchical clustering on distance 1 - |rho|, cut at
    1 - CLUSTER_ABS_RHO.
    """
    X = np.asarray(X_codes, float)
    n, p = X.shape
    if p == 1:
        return {names[0]: 1}
    R = np.eye(p)
    for a in range(p):
        for b in range(a + 1, p):
            ok = np.isfinite(X[:, a]) & np.isfinite(X[:, b])
            if ok.sum() < 20 or np.nanstd(X[ok, a]) == 0 or np.nanstd(X[ok, b]) == 0:
                r = 0.0
            else:
                r = spearmanr(X[ok, a], X[ok, b]).statistic
                r = 0.0 if not np.isfinite(r) else abs(float(r))
            R[a, b] = R[b, a] = r
    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    lab = fcluster(Z, t=1.0 - CLUSTER_ABS_RHO, criterion="distance")
    return {names[j]: int(lab[j]) for j in range(p)}


def neff_cluster(marginal_aucs: dict[str, float], clusters: dict[str, int], n1: int, n0: int) -> float:
    """Cluster-adjusted N_eff on DENOISED lifts (stratifier.lifts_dn, same threshold as the primary
    stratifier): cluster lift = the largest denoised member lift; N_eff over clusters.
    NaN when no feature exceeds the denoising threshold."""
    names = list(marginal_aucs)
    lifts = stratifier.lifts_dn([marginal_aucs[f] for f in names], n1, n0)
    best: dict[int, float] = {}
    for f, lift in zip(names, lifts):
        c = clusters[f]
        best[c] = max(best.get(c, 0.0), float(lift))
    return _neff_from_lifts(list(best.values()))


# --------------------------------------------------------------------------------------------------
# 2. conditional N_eff from the reference logistic regression
# --------------------------------------------------------------------------------------------------

def feature_contributions(coef: np.ndarray, Xt_heldout: np.ndarray, groups: list[str]) -> dict[str, np.ndarray]:
    """Per-original-feature contribution to the linear predictor on held-out rows.

    coef       : (m,) coefficients of the fitted reference LR (intercept excluded), m transformed columns
    Xt_heldout : (n_held, m) held-out rows AFTER the fitted preprocessing of that fold (imputation,
                 scaling, one-hot), i.e. exactly what the LR multiplies by coef
    groups     : length-m list giving the ORIGINAL feature name of each transformed column
    """
    coef = np.asarray(coef, float).ravel()
    Xt = np.asarray(Xt_heldout, float)
    if Xt.shape[1] != len(coef) or len(groups) != len(coef):
        raise ValueError("coef, Xt_heldout and groups must agree in length")
    out: dict[str, np.ndarray] = {}
    for g in dict.fromkeys(groups):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out[g] = Xt[:, idx] @ coef[idx]
    return out


def neff_conditional(per_fold: list[dict[str, np.ndarray]], feature_names: list[str]) -> float:
    """per_fold: one dict per outer fold from feature_contributions (the five frozen folds).
    Contributions are stacked over the held-out folds; lift_j = population SD of the stacked
    contributions of feature j (a feature absent from every fold's design, e.g. dropped as constant,
    has lift 0); N_eff_cond = (sum lift)^2 / sum lift^2."""
    lifts = []
    for f in feature_names:
        parts = [d[f] for d in per_fold if f in d]
        lifts.append(float(np.std(np.concatenate(parts))) if parts else 0.0)
    return _neff_from_lifts(lifts)


# --------------------------------------------------------------------------------------------------
# 3. medical-subset candidate rule
# --------------------------------------------------------------------------------------------------

MEDICAL_TERMS = [
    r"patient", r"clinic", r"hospital", r"medic", r"diagnos", r"disease", r"cancer", r"tumou?r",
    r"carcinom", r"diabet", r"cardi", r"heart", r"surg", r"liver", r"hepat", r"kidney", r"renal",
    r"thyroid", r"breast", r"lung", r"blood", r"mortality", r"survival", r"health", r"biomed",
    r"gene expression", r"microarray", r"gse\d{3,}", r"stroke", r"dementia", r"alzheimer", r"parkinson",
    r"psychiatr", r"schizo", r"cholesterol", r"obes", r"pregnan", r"birth", r"infect", r"covid",
    r"spect\b", r"ecg", r"eeg", r"mri", r"pathol", r"epidemiol",
]
_MED = re.compile("|".join(MEDICAL_TERMS), flags=re.IGNORECASE)


def medical_candidate(name: str, description: str | None, tags: list[str] | None) -> list[str]:
    """Return the matched terms (empty list = not a candidate). Candidates are then reviewed by hand
    and labelled 'clinical' (individual-level clinical or health records), 'biomedical' (laboratory,
    imaging-derived or omics measurements on human subjects) or 'non-medical'; the label and a
    one-line reason are frozen before registration."""
    text = " ".join([name or "", description or "", " ".join(tags or [])])
    return sorted({m.group(0).lower() for m in _MED.finditer(text)})
