# Deviations decided during the Phase 1 run

Registration: OSF, DOI 10.17605/OSF.IO/GAFDR; repository commit
`68d5d00916ca2af60fdda4e7272bd416611aa531`. Deviations D1 and D2 are in
`phase1_section0/DEVIATIONS.md` (commit `b7ba954`). The frozen driver is unchanged: its manifest
SHA-256 is `136c1e4207c5168be34288ac4846aae147bc4234ba4e566f23d8adcdfd0cc68c`.

Both deviations below were decided on 2026-09-29, from run times only. Model outputs of completed
units existed in the checkpoints, but none had been aggregated or examined when the decisions were
taken and committed. Each deviation will be reported in the manuscript, as Section 11 of the plan
requires. The timing files cited are in `evidence/`; none of them contains a performance quantity.

## D3. Two families are excluded because EBM cannot complete them in practical time

**Families.** albert (OpenML 44538–44542) and Amazon employee access (OpenML 44708–44712). Each is
one family of five 2,000-row subsamples of a single OpenML dataset, made by the same subsampling
code. All ten datasets are in the target region: albert has four in T3 and one in T2, and Amazon
has all five in T1.

**What they share.** Their features are anonymised identifier-like categorical columns with
thousands of levels in only 2,000 rows. albert has 52 categorical columns out of 70, with 25,869 to
26,099 levels in total per dataset. Amazon has 8 categorical columns out of 8, with 2,788 to 2,898
levels in total. The next target-region family by total levels, kick (15 categorical columns, about
1,050 levels), ran normally: about 0.9 hours per M1 unit.

**Observed run times** (`evidence/phase1_diag_timeout.zip`, `evidence/m1_timing_check.zip`).
EBM was run with its registered library defaults (interpret 0.7.8, `max_rounds` 50,000,
early stopping).
- Amazon, M1. All 25 conventional-model units exceeded the D2 unit limit (7,573 s) twice. A single
  EBM fit on fold 0 took 5,031 s. The other five conventional models took 1.0 to 14.5 s on the same
  data. On an M2 variant (`dispn_0.85_r0`, fold 0), one EBM fit took 708 s and 1,217 s in two runs.
- albert, M1. Four of the 25 units finished, in 8.5 to 11.9 hours, against a projection of 1.4
  hours. The other 21 exceeded the unit limit (51,328 s) on their first attempt.
- albert, M2. Three single EBM fits on 44541 (`conc_0.85_r0` fold 0, `dispn_0.85_r0` folds 0 and 3)
  each exceeded a 3-hour cap. A unit needs six EBM fits (five of them on 80% of the training rows),
  so an albert M2 unit would take about 15 hours or more. That is at or above its limit of
  56,481 s (15.7 h), or 60,941 s for the dispersed-noise arm. Timing table `albert_m2_timing.csv`,
  SHA-256 `dfc5a4c19867f8511b6364a9286c8a506f4835329e3394b44dd63dcb7e5d030b`.

**Cost.** The families have 975 + 975 M2 and 500 + 375 M4 conventional-model units. Running the
albert M2 units alone would take at least about 12,000 core-hours, about one week of both CPU
instances, with repeated time-outs likely.

**Change.** Both families are excluded from every module and every analysis.
- Their conventional-model units that had no checkpoint were given terminal records without
  fitting any model, written with the frozen driver's `failed_unit_payload`. There are 21 M1 units
  (`error_type` `stopped_d3`: their first attempt had exceeded the limit and the repeat attempt was
  stopped), and 1,950 M2 and 875 M4 units (`error_type` `not_run_d3`). By host: 214 has 11, 1,170
  and 525; 686 has 10, 780 and 350.
- Their M1 units that had already failed by time-out keep that record and are not rerun under D4.
- Their foundation-model units run on the GPU as scheduled. They enter no analysis, because there
  is no conventional comparator.

**Size of the loss.** 2 of the 86 target-region families, and 10 of the 238 target-region
datasets (4.2%), below the 10% threshold of Section 6.8. Amazon is 1 of the 27 T1 families.

**Supplementary runs.** The main analyses exclude both families. If either family is later run in
full without a time limit, every family so run is reported in full as a supplementary analysis,
with the primary and key secondary analyses repeated including it.

**Limitation.** EBM with default settings may not complete on data with very high-cardinality
categorical features. This is reported as a limitation of the comparator set.

## D4. Units that failed by time-out are rerun without a time limit

**Registered.** Under D2, a unit that exceeds its time limit twice is recorded as failed. Under
Section 6.8, the variant is then excluded from every analysis that needs that model.

**Problem.** EBM run times vary strongly between folds and variants of the same dataset, so
time-outs are not a property of the dataset alone.
- 920, M1: three folds took about 26 minutes each, one took 6.0 hours, and one exceeded its
  8.1-hour limit twice.
- 59, M1: two folds took about 25 minutes each, two took 2.6 to 2.8 hours, and one exceeded its
  limit.
- In M2, units of datasets whose M1 units took 11 to 13 minutes exceeded limits of 6 to 9 hours:
  1065 (`conc_0.95_r0`), 1443 (`conc_0.75_r0`, `conc_0.95_r0`, `dispn_0.95_r0`) and 1467
  (`dispn_0.95_r0`).

Four of these five M2 cases are at the highest injection level. Leaving such units as failures
would remove cells unevenly across levels and arms of the injection contrast.

**Change.**
1. After the main run, every unit recorded as failed by time-out is rerun once, without a per-unit
   time limit. Failed by time-out means `error_type` `timeout` on its last attempt. The two D3
   families are excluded from this.
2. The rerun uses the frozen driver's own worker function, with the same code path and the same
   recorded environment as the scheduler. The frozen driver files are not modified. The small
   runner that calls it will be committed, with its SHA-256, before it is used.
3. The original failure records are archived, not deleted, and every rerun is listed with its run
   time.
4. The confirmatory analyses use the rerun results.
5. A unit that still fails, or that cannot be completed before the compute runs out, is handled
   under Section 6.8 and listed.
6. The same applies to any foundation-model unit that fails by time-out.

**Why.** It restores the registered handling of slow units, which had no time limit, and it keeps
the analysis sets from depending on where EBM happened to run long.
