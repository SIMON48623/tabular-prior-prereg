"""Frozen stratifier definitions (preregistration v2, Section 4).

primary   : log N_eff_dn  - denoised effective number of contributing features
sensitivity: log N_eff_raw - the v1 definition (no noise correction)

Marginal AUROCs are the frozen Phase 0b values (out-of-fold: direction chosen on the training fold,
held-out values mapped through the training fold's empirical distribution, pooled over the five
folds). No other quantity is used.

Why denoise: under the null of no association, a held-out marginal AUROC fluctuates around 0.5 with
standard error SE0 = sqrt((n0 + n1 + 1) / (12 n0 n1)) (Hanley & McNeil 1982). With lift = max(AUC - 0.5, 0)
every null feature contributes a positive lift about half the time, so the raw N_eff grows with the
number of null features and shrinks with n. In the informal exploration that preceded this file
(code/exploration, step 01), a single informative feature among 32 gave raw N_eff between about 3.4
(n = 3000) and 10.6 (n = 300). Subtracting a
one-sided 95% null quantile from every lift removes most of this inflation while preserving the
ranking of truly dispersed signal.
"""
from __future__ import annotations

import math

import numpy as np

Z_DENOISE = 1.645


def se0(n1: int, n0: int) -> float:
    """Null standard error of an AUROC with n1 positives and n0 negatives (Hanley & McNeil)."""
    return math.sqrt((n0 + n1 + 1) / (12.0 * n0 * n1))


def lifts_raw(aucs) -> np.ndarray:
    return np.maximum(np.asarray(list(aucs), float) - 0.5, 0.0)


def lifts_dn(aucs, n1: int, n0: int, z: float = Z_DENOISE) -> np.ndarray:
    return np.maximum(np.asarray(list(aucs), float) - 0.5 - z * se0(n1, n0), 0.0)


def hill2(lifts) -> float:
    """(sum l)^2 / sum l^2; NaN when every lift is zero."""
    lifts = np.asarray(lifts, float)
    s2 = float((lifts ** 2).sum())
    return float(lifts.sum() ** 2 / s2) if s2 > 0 else float("nan")


def neff_raw(aucs) -> float:
    return hill2(lifts_raw(aucs))


def neff_dn(aucs, n1: int, n0: int) -> float:
    """Denoised N_eff. Undefined (NaN) when no feature exceeds 0.5 + 1.645 SE0; such datasets are
    excluded from every analysis that uses N_eff_dn (declared in the plan)."""
    return hill2(lifts_dn(aucs, n1, n0))
