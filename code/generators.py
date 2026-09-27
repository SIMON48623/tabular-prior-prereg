"""Frozen data manipulations for Phase 1 (preregistration v2).

This file is registered together with the preregistration. The executor must import it unchanged and
must not re-implement any function in it. Every random draw is derived from `seed_for(...)`, so the
same call returns the same array on any machine with the same numpy major version.

Contents
--------
1. seed_for             deterministic 32-bit seeds from labelled parts
2. injection_block      the injected columns for the injection arm (Section 5.4 of the plan)
3. removal_plan         top-k and random-k removal sets from the frozen marginal AUROCs (Section 5.5)
4. subsample_indices    stratified training-fold subsamples for the sample-size arm (Section 5.6)
5. synth_draw           synthetic mechanism generator (Section 5.7), using the frozen calibration table
6. expected_auroc       tie-aware expected AUROC of a score against Bernoulli(p) labels
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

# --------------------------------------------------------------------------------------------------
# 1. seeds
# --------------------------------------------------------------------------------------------------

def seed_for(*parts) -> int:
    """32-bit seed from SHA-256 of the '|'-joined string representation of `parts`."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


# --------------------------------------------------------------------------------------------------
# 2. injection arm
# --------------------------------------------------------------------------------------------------

INJECT_LEVELS = (0.65, 0.75, 0.85, 0.95)   # target marginal AUROC of the concentrated column
INJECT_K = 8                               # columns in the signal block
INJECT_EXTRA_NOISE = 7                     # extra pure-noise columns in the 'dispersed_noise' arm
INJECT_REPS = 3                            # independent injection replicates per dataset
INJECT_ARMS = ("noise", "concentrated", "dispersed", "dispersed_noise")


def injection_separation(level: float) -> float:
    """Mean shift d (in noise-SD units) giving marginal AUROC `level` for z = d*y + N(0,1):
    AUROC = Phi(d / sqrt(2))."""
    return math.sqrt(2.0) * norm.ppf(level)


def injection_block(y, openml_id, arm: str, level: float | None, rep: int = 0):
    """Return (Z, names): columns to append to the cleaned feature matrix.

    y        : 0/1 labels of the cleaned dataset (positive = Phase 0b minority class), full dataset
               order (row_index 0..n-1). Injection is done once on the full dataset, before folds.
    arm      : 'noise'           8 pure-noise columns (level must be None)
               'concentrated'    8 columns: one carries separation d, seven are pure noise (= F)
               'dispersed'       8 columns, each carrying separation d / sqrt(8)       (column-matched)
               'dispersed_noise' the 8 'dispersed' columns plus 7 further pure-noise columns
                                 (15 columns; noise-matched to 'concentrated')
    level    : one of INJECT_LEVELS (ignored for 'noise').
    rep      : injection replicate, 0 .. INJECT_REPS - 1 (independent noise draws).

    Information matching: all signal-bearing columns are conditionally independent unit-variance
    Gaussians given y, so the Bayes AUROC of the injected block is Phi(D / sqrt(2)) with
    D^2 = sum_j d_j^2 = d^2 in every signal arm. 'concentrated' and 'dispersed_noise' also contain the
    same 7 pure-noise columns; they differ in how the information is spread (1 strong column versus
    8 weak ones) and in total column count (8 versus 15).

    Common random numbers: within a dataset and replicate, all arms and levels are built from the same
    draws: the signal-column noise matrix E (n x 8), the pure-noise matrix F (n x 7), the position
    `pos` of the concentrated column and the column order of the 15-column arm.
      noise           = E
      concentrated    = F with E[:, pos] + d*y inserted at position pos (its 7 pure-noise columns
                        are exactly the 7 pure-noise columns of 'dispersed_noise')
      dispersed       = E + (d / sqrt(8)) * y
      dispersed_noise = [E + (d / sqrt(8)) * y, F], columns permuted by the seeded order
    """
    y = np.asarray(y).astype(float)
    if set(np.unique(y)) - {0.0, 1.0}:
        raise ValueError("y must be coded 0/1")
    if rep not in range(INJECT_REPS):
        raise ValueError(f"rep must be in 0..{INJECT_REPS - 1}")
    n = len(y)
    E = np.random.default_rng(seed_for("inject-noise", openml_id, rep)).standard_normal((n, INJECT_K))
    F = np.random.default_rng(seed_for("inject-extra-noise", openml_id, rep)).standard_normal((n, INJECT_EXTRA_NOISE))
    pos = int(np.random.default_rng(seed_for("inject-position", openml_id, rep)).integers(INJECT_K))
    order15 = np.random.default_rng(seed_for("inject-order15", openml_id, rep)).permutation(INJECT_K + INJECT_EXTRA_NOISE)
    if arm == "noise":
        if level is not None:
            raise ValueError("noise arm takes level=None")
        return E.copy(), [f"inj_{j + 1}" for j in range(INJECT_K)]
    if level not in INJECT_LEVELS:
        raise ValueError(f"level must be one of {INJECT_LEVELS}")
    d = injection_separation(level)
    if arm == "concentrated":
        Z = np.insert(F, pos, E[:, pos] + d * y, axis=1)
    elif arm == "dispersed":
        Z = E + (d / math.sqrt(INJECT_K)) * y[:, None]
    elif arm == "dispersed_noise":
        Z = np.column_stack([E + (d / math.sqrt(INJECT_K)) * y[:, None], F])[:, order15]
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return Z, [f"inj_{j + 1}" for j in range(Z.shape[1])]


def injection_expected_lifts(arm: str, level: float | None) -> list[float]:
    """Population marginal lifts (AUROC - 0.5) of the injected columns, for the manipulation check
    (pure-noise columns have lift 0)."""
    if arm == "noise":
        return [0.0] * INJECT_K
    d = injection_separation(level)
    if arm == "concentrated":
        return [level - 0.5] + [0.0] * (INJECT_K - 1)
    per = norm.cdf(d / math.sqrt(INJECT_K) / math.sqrt(2.0)) - 0.5
    if arm == "dispersed":
        return [per] * INJECT_K
    if arm == "dispersed_noise":
        return [per] * INJECT_K + [0.0] * INJECT_EXTRA_NOISE
    raise ValueError(arm)


# --------------------------------------------------------------------------------------------------
# 3. removal arm
# --------------------------------------------------------------------------------------------------

REMOVAL_K = (1, 3)
REMOVAL_RANDOM_DRAWS = 5


def removal_plan(openml_id, marginal_aucs: dict[str, float]) -> dict:
    """Removal sets for one dataset from its frozen marginal AUROCs.

    top_k      : the k features with the highest marginal AUROC (ties broken by feature name, ascending)
    random_k   : REMOVAL_RANDOM_DRAWS draws, each a set of k features sampled without replacement from
                 the features outside top_k. Draws are independent of each other.
    The k = 3 arm requires at least 6 features (so that a random draw of 3 exists outside the top 3).
    """
    feats = sorted(marginal_aucs, key=lambda f: (-marginal_aucs[f], f))
    p = len(feats)
    plan = {}
    for k in REMOVAL_K:
        if p - k < k:
            plan[k] = None
            continue
        top = feats[:k]
        rest = sorted(feats[k:])
        rng = np.random.default_rng(seed_for("removal-random", openml_id, k))
        draws = [sorted(rng.choice(rest, size=k, replace=False).tolist())
                 for _ in range(REMOVAL_RANDOM_DRAWS)]
        plan[k] = {"top": top, "random": draws}
    return plan


def n_eff(aucs) -> float:
    lift = np.maximum(np.asarray(list(aucs), float) - 0.5, 0.0)
    s2 = float((lift ** 2).sum())
    return float(lift.sum() ** 2 / s2) if s2 > 0 else float("nan")


# --------------------------------------------------------------------------------------------------
# 4. sample-size arm
# --------------------------------------------------------------------------------------------------

SUBSAMPLE_SIZES = (100, 250, 500, 1000)
SUBSAMPLE_REPS = 5
SUBSAMPLE_MIN_EXPECTED_POSITIVES = 10


def subsample_eligible(size: int, n_train: int, minority_rate: float) -> bool:
    """A size is run for a dataset only if it is strictly smaller than the smallest outer training
    fold and the expected number of positives is at least 10."""
    return size < n_train and size * minority_rate >= SUBSAMPLE_MIN_EXPECTED_POSITIVES


def subsample_indices(y_train, openml_id, fold: int, size: int, rep: int) -> np.ndarray:
    """Stratified subsample (positions within the outer training fold) of the given size.

    Class counts: round(size * positive share of the training fold), at least 2 per class; the
    remaining rows are drawn from the other class. Returned positions are sorted ascending.
    """
    y_train = np.asarray(y_train).astype(int)
    rng = np.random.default_rng(seed_for("subsample", openml_id, fold, size, rep))
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    n_pos = int(round(size * len(pos_idx) / len(y_train)))
    n_pos = min(max(n_pos, 2), len(pos_idx), size - 2)
    n_neg = size - n_pos
    if n_neg > len(neg_idx):
        raise ValueError("size exceeds available negatives")
    take = np.concatenate([rng.choice(pos_idx, n_pos, replace=False),
                           rng.choice(neg_idx, n_neg, replace=False)])
    return np.sort(take)


# --------------------------------------------------------------------------------------------------
# 5. synthetic mechanism generator
# --------------------------------------------------------------------------------------------------

SYN_P = 32
SYN_K = (1, 2, 4, 8, 16, 32)
SYN_GEN = ("linear", "threshold")
SYN_RHO = (0.0, 0.3)
SYN_AUC = (0.70, 0.80)
SYN_PREVALENCE = 0.30
SYN_N_TRAIN = (100, 300, 1000, 3000)
SYN_N_TEST = 5000
SYN_REPS = 20
CALIB_FILE = Path(__file__).with_name("synth_calibration.json")


def _g(gen: str, X: np.ndarray) -> np.ndarray:
    if gen == "linear":
        return X
    if gen == "threshold":
        return np.where(X > 0.0, 1.0, -1.0)
    raise ValueError(gen)


def synth_latent(gen: str, k: int, rho: float, n: int, rng: np.random.Generator):
    """Draw features and the un-scaled latent score. Features are equicorrelated N(0,1) with
    correlation rho; the first k columns (before permutation) are informative with equal weight."""
    z0 = rng.standard_normal((n, 1))
    Z = rng.standard_normal((n, SYN_P))
    X = math.sqrt(rho) * z0 + math.sqrt(1.0 - rho) * Z
    eta0 = _g(gen, X[:, :k]).sum(axis=1) / math.sqrt(k)
    return X, eta0


def _load_calibration() -> dict:
    return json.loads(CALIB_FILE.read_text())


def synth_draw(gen: str, k: int, rho: float, auc: float, n: int, rep: int, split: str,
               calibration: dict | None = None):
    """Draw one synthetic dataset. Returns (X, y, p_true, informative_columns).

    split is 'train' or 'test'; the train and test draws for the same cell and replicate use
    different seeds. Column order is permuted once per (cell, rep), independently of n, so it is shared
    by the training sets of every size and by the test set. The test set is drawn with n = SYN_N_TEST
    and is therefore identical for every training size of a cell and replicate.
    """
    cal = calibration or _load_calibration()
    key = f"{gen}|k={k}|rho={rho}|auc={auc}"
    s, b0 = cal[key]["s"], cal[key]["b0"]
    rng = np.random.default_rng(seed_for("synth", gen, k, rho, auc, n, rep, split))
    X, eta0 = synth_latent(gen, k, rho, n, rng)
    p_true = 1.0 / (1.0 + np.exp(-(b0 + s * eta0)))
    y = (rng.random(n) < p_true).astype(int)
    perm = np.random.default_rng(seed_for("synth-perm", gen, k, rho, auc, rep)).permutation(SYN_P)
    X = X[:, perm]
    informative = sorted(int(np.flatnonzero(perm == j)[0]) for j in range(k))
    return X, y, p_true, informative


# --------------------------------------------------------------------------------------------------
# 6. expected AUROC (tie-aware, no label sampling)
# --------------------------------------------------------------------------------------------------

def expected_auroc(score: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> float:
    """AUROC of `score` against labels Y ~ Bernoulli(p), in expectation over Y:
    sum_{i,j} w_i p_i w_j (1 - p_j) [s_i > s_j, ties 1/2] / (sum w p)(sum w (1 - p)).
    `weight` are probability masses when score/p describe a discretised distribution (default 1)."""
    score = np.asarray(score, float)
    p = np.asarray(p, float)
    w = np.ones_like(p) if weight is None else np.asarray(weight, float)
    order = np.argsort(score, kind="mergesort")
    s, pw, nw = score[order], (w * p)[order], (w * (1.0 - p))[order]
    uniq, start = np.unique(s, return_index=True)
    P = np.add.reduceat(pw, start)
    N = np.add.reduceat(nw, start)
    N_below = np.concatenate([[0.0], np.cumsum(N)[:-1]])
    num = float((P * (N_below + 0.5 * N)).sum())
    return num / (pw.sum() * nw.sum())
