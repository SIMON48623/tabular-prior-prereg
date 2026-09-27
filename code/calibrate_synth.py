"""Calibrate the synthetic generator (run once before registration; output is frozen).

For each cell (generator, k, rho, target Bayes AUROC) find scale s and intercept b0 such that the
prevalence is 0.30 and the Bayes AUROC (AUROC of the true probability) equals the target.

The latent score eta0 = sum_{j<=k} g(x_j) / sqrt(k) has a one-dimensional distribution that is
computed exactly by quadrature, so the calibration has no Monte Carlo error:
  linear    : eta0 ~ N(0, 1 + (k - 1) rho)
  threshold : eta0 = (2M - k) / sqrt(k), M = number of positive x_j among k equicorrelated normals;
              P(M = m) = E_z0[ Binom(m; k, Phi(sqrt(rho) z0 / sqrt(1 - rho))) ].
A Monte Carlo check (200,000 draws from the actual generator) then records the realised Bayes AUROC,
the prevalence, the population marginal AUROC of every column and the population N_eff.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import binom, norm

sys.path.insert(0, str(Path(__file__).parent))
import generators as G  # noqa: E402

N_CHECK = 200_000


def eta_distribution(gen, k, rho):
    if gen == "linear":
        sd = math.sqrt(1.0 + (k - 1) * rho)
        grid = np.linspace(-9 * sd, 9 * sd, 40001)
        w = norm.pdf(grid, scale=sd)
        return grid, w / w.sum()
    nodes, wts = np.polynomial.hermite_e.hermegauss(200)
    wts = wts / wts.sum()
    if rho == 0.0:
        q = np.full_like(nodes, 0.5)
    else:
        q = norm.cdf(math.sqrt(rho) * nodes / math.sqrt(1.0 - rho))
    m = np.arange(k + 1)
    pm = (binom.pmf(m[:, None], k, q[None, :]) * wts[None, :]).sum(axis=1)
    return (2 * m - k) / math.sqrt(k), pm / pm.sum()


def solve_b0(eta, w, s, prev):
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (w * expit(mid + s * eta)).sum() < prev:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate(gen, k, rho, target):
    eta, w = eta_distribution(gen, k, rho)

    def auc_at(s):
        b0 = solve_b0(eta, w, s, G.SYN_PREVALENCE)
        return G.expected_auroc(eta, expit(b0 + s * eta), w), b0

    lo, hi = 0.0, 60.0
    if auc_at(hi)[0] < target:
        raise RuntimeError(f"target {target} unreachable for {gen} k={k} rho={rho}")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if auc_at(mid)[0] < target else (lo, mid)
    s = 0.5 * (lo + hi)
    a, b0 = auc_at(s)
    return {"s": s, "b0": b0, "bayes_auroc_exact": a}


def mc_check(gen, k, rho, auc, cal):
    rng = np.random.default_rng(G.seed_for("calibration-check", gen, k, rho, auc))
    X, eta = G.synth_latent(gen, k, rho, N_CHECK, rng)
    p = expit(cal["b0"] + cal["s"] * eta)
    marg = []
    for j in range(G.SYN_P):
        m = G.expected_auroc(X[:, j], p)
        marg.append(max(m, 1.0 - m))
    return {"bayes_auroc_mc": G.expected_auroc(eta, p), "prevalence_mc": float(p.mean()),
            "marginal_auroc": marg, "n_eff_population": G.n_eff(marg)}


if __name__ == "__main__":
    out = {}
    for gen in G.SYN_GEN:
        for k in G.SYN_K:
            for rho in G.SYN_RHO:
                for auc in G.SYN_AUC:
                    key = f"{gen}|k={k}|rho={rho}|auc={auc}"
                    cal = calibrate(gen, k, rho, auc)
                    cal.update(mc_check(gen, k, rho, auc, cal))
                    out[key] = cal
                    print(f"{key:34s} s={cal['s']:.4f} b0={cal['b0']:+.4f} exact={cal['bayes_auroc_exact']:.5f} "
                          f"mc={cal['bayes_auroc_mc']:.4f} prev={cal['prevalence_mc']:.4f} "
                          f"Neff_pop={cal['n_eff_population']:.2f} max_marg={max(cal['marginal_auroc']):.3f}", flush=True)
    G.CALIB_FILE.write_text(json.dumps(out, indent=1))
