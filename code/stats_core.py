"""Core estimators for the Phase 1 confirmatory analyses (frozen with the preregistration).

Only numpy and scipy are used so that the file runs unchanged in any environment.

Conventions
-----------
* The unit of independence is the provenance family.
* Family weights: each dataset i in family f receives w_i = 1 / m_f, where m_f is the number of
  datasets of family f in the analysis set actually used (after exclusions). Every family therefore
  contributes a total weight of 1.
* Standard errors are cluster-robust (CR1) with clustering on family; inference uses a t reference
  distribution with G - 1 degrees of freedom, G = number of families in the analysis set.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def family_weights(family) -> np.ndarray:
    fam = np.asarray(family)
    _, inv, counts = np.unique(fam, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def hierarchical_weights(family, dataset) -> np.ndarray:
    """Weights for designs with several rows per dataset (sample-size arm, S5): row r of dataset i in
    family f receives 1 / (m_f * L_i), where m_f is the number of distinct datasets of family f and
    L_i the number of rows of dataset i among the rows supplied. Every family carries total weight 1
    and, within a family, every dataset carries equal weight."""
    fam = np.asarray(family)
    ds = np.asarray(dataset)
    w = np.empty(len(fam), float)
    for f in np.unique(fam):
        in_f = fam == f
        dsets = np.unique(ds[in_f])
        for d in dsets:
            idx = in_f & (ds == d)
            w[idx] = 1.0 / (len(dsets) * idx.sum())
    return w


def wls_cluster(y, X, family, weights=None, add_const=True, dataset=None):
    """Weighted least squares with CR1 cluster-robust covariance.

    Weights, in order of precedence:
      weights given  -> used as supplied (caller is responsible for them);
      dataset given  -> hierarchical_weights(family, dataset), computed on the rows actually analysed;
      otherwise      -> family_weights(family), computed on the rows actually analysed.
    Rows with a non-finite outcome, covariate or supplied weight are dropped before weights are
    computed, so that every family keeps total weight 1 in the default and hierarchical cases.

    Returns dict with beta, se, t, df, p (two-sided), ci95 (two-sided), cov, names order = columns of X
    (constant first when add_const=True).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
    fam = np.asarray(family)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if weights is not None:
        w_all = np.asarray(weights, float)
        ok &= np.isfinite(w_all)
    y, X, fam = y[ok], X[ok], fam[ok]
    # default and hierarchical weights are computed on the rows actually analysed
    if weights is not None:
        w = w_all[ok]
    elif dataset is not None:
        w = hierarchical_weights(fam, np.asarray(dataset)[ok])
    else:
        w = family_weights(fam)
    n, k = X.shape
    XtW = X.T * w
    bread = np.linalg.inv(XtW @ X)
    beta = bread @ (XtW @ y)
    e = y - X @ beta
    groups = np.unique(fam)
    G = len(groups)
    meat = np.zeros((k, k))
    for g in groups:
        idx = fam == g
        s = (X[idx].T * w[idx]) @ e[idx]
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if n > k else G / (G - 1)
    cov = c * bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    df = G - 1
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    q = stats.t.ppf(0.975, df)
    return {
        "beta": beta, "se": se, "t": t, "df": df, "p": p,
        "ci95": np.column_stack([beta - q * se, beta + q * se]),
        "cov": cov, "n": n, "G": G,
    }


def weighted_mean_cluster(y, family, weights=None):
    """Family-weighted mean with CR1 cluster-robust SE (intercept-only WLS)."""
    r = wls_cluster(y, np.empty((len(np.asarray(y)), 0)), family, weights=weights, add_const=True)
    return {"mean": r["beta"][0], "se": r["se"][0], "df": r["df"], "G": r["G"], "n": r["n"],
            "p": r["p"][0], "ci95": r["ci95"][0]}


def one_sided_upper(mean, se, df, alpha=0.025):
    return mean + stats.t.ppf(1 - alpha, df) * se


def one_sided_lower(mean, se, df, alpha=0.025):
    return mean - stats.t.ppf(1 - alpha, df) * se


def one_sided_p(estimate, se, df, null=0.0, direction="greater") -> float:
    """One-sided p-value of a t statistic with df degrees of freedom.
    direction='greater': H0 estimate <= null (evidence for estimate > null);
    direction='less'   : H0 estimate >= null (evidence for estimate < null)."""
    t = (estimate - null) / se
    if direction == "greater":
        return float(stats.t.sf(t, df))
    if direction == "less":
        return float(stats.t.cdf(t, df))
    raise ValueError(direction)


def holm(pvals: dict[str, float], alpha=0.025) -> dict[str, dict]:
    """Holm step-down. Returns adjusted p-values and reject flags."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        adj = min(1.0, (m - i) * pvals[k])
        running = max(running, adj)
        out[k] = {"p": pvals[k], "p_holm": running, "reject": running < alpha}
    return out


def load_cutpoints(path) -> tuple[float, float]:
    """Read the frozen cutpoints (cutpoints_v2.json) of log N_eff_dn."""
    import json
    c = json.loads(open(path).read())
    return float(c["c1_log_neff_dn"]), float(c["c2_log_neff_dn"])


def tertile(log_neff, c1: float, c2: float):
    """Tertile assignment with frozen cutpoints: T1 < c1 <= T2 < c2 <= T3; NaN -> 'undefined'."""
    x = np.asarray(log_neff, float)
    out = np.where(x < c1, "T1", np.where(x < c2, "T2", "T3"))
    return np.where(np.isnan(x), "undefined", out)
