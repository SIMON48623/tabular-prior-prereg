# Informal pre-registration explorations (record only)

These scripts and outputs are the informal explorations that led to the denoised stratifier
(N_eff_dn) and to the threshold z = 1.645. They were run on 2026-09-26 between 14:05 and 14:10 UTC,
before `stratifier.py`, `construct_check.py` and `threshold_choice.py` were written. They are kept
so that the account in Section 4.3 of the preregistration can be checked.

- The commands (`*.command.sh`) and printed outputs (`*.output.txt`) were recovered verbatim from the
  session log. The helper scripts they call (`construct.py`, `neff_est.py`, `neff_est2.py`,
  `real_dn.py`) are the files as they were at that time.
- Steps 01–03 used synthetic data from the frozen generator only. The marginal AUROCs in these
  steps ranked held-out values within the held-out fold, a cruder emulation of Phase 0b than the one
  later used in `construct_check.py` (training-fold ECDF mapping).
- Steps 04–05 computed the denoised measure on the frozen Phase 0b marginal AUROCs of the pool for
  z in {0, 1.0, 1.645, 1.96} and inspected its correlations with log p, log n, auc_lr and minority
  rate, and the resulting tertiles, before z = 1.645 was frozen in `stratifier.py`. With z = 1.0 the
  correlation with auc_lr (about +0.23) would have met Phase 0b's |rho| <= 0.25 rule; z = 1.645 was
  kept on the synthetic criterion of step 03. No model other than the Phase 0b reference logistic
  regression, and no model-comparison quantity, was involved.
- The comparison in step 03 (mean absolute error against log k, rank correlation with k, and
  inflation from p = 12 to p = 32, for z in {0, 1.0, 1.645, 1.96, 2.576} and a hard threshold) is
  the informal comparison in which z = 1.645 was chosen. The frozen `threshold_choice.py` repeats it
  on the construct-check draws.

These files are not frozen analysis code and are not used by any registered analysis.

The files are kept exactly as recorded. Paths in them (for example `/home/claude/...` and
`/tmp/claude-0/.../scratchpad`) refer to the development machine, and one output file contains a line
printed by the shell environment; neither affects the results.
