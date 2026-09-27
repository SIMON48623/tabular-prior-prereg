# Preregistration

## Dispersion of predictive signal and the value of tabular foundation models: a preregistered study combining within-dataset intervention, cross-dataset observation and synthetic mechanism experiments

**Version** 2.0 (supersedes the unregistered draft v1.0 of 2026-09-15; changes listed in Section 0)
**Date** 2026-09-27
**Author** Yuzhang Wu (Beijing University of Posts and Telecommunications / Queen Mary University of London)
**Registration platform** OSF, with a time-stamped public Git commit (Section 12)
**Status at registration** No model other than the reference logistic regression has been fitted to any dataset in the pool, and no model-comparison quantity exists. Section 10 lists everything that has been observed.

Every number that depends on the dataset pool or on simulation is computed from the frozen files by `fill_prereg.py`: the final pool (`merge_pool.py`), its stratifiers and cutpoints (`build_frozen_stratifier.py`), removal plan and manipulation checks (`build_frozen_plans.py`), and power (`power_sim.py`). Phase 0c, which completed the pool before registration, fitted no model other than the reference logistic regression on pool data.

---

## 0. Changes from the unregistered draft v1

Draft v1 was never registered and no experiment was run under it. Every change below was made without any model-comparison result.

| # | v1 | v2 | Reason |
|---|---|---|---|
| 1 | TabPFN, version unspecified | TabPFN-3.5 (primary), TabICLv2 and TabPFN-2, each pinned to a checkpoint trained on synthetic data only | Newer models were released; generality across models and developers |
| 2 | Comparator: CatBoost | Comparator: the conventional model selected by inner cross-validation from six; plus a key secondary test against the better of the two additive models (LR, EBM) | A fixed tree comparator cannot separate a foundation-model effect from "smooth additive models beat trees" |
| 3 | Pooled out-of-fold AUROC | Mean of the per-fold AUROCs | The selected comparator can differ between folds, so its scores cannot be pooled |
| 4 | Unweighted OLS, covariate log p | Family-weighted least squares; covariates log p, log n, auc_lr | Family is the unit of independence; see item 6 for auc_lr |
| 5 | Observational test only | Adds a within-dataset intervention (primary, tested first) and sample-size, synthetic and removal arms | An association across datasets cannot separate dispersion from covarying dataset properties |
| 6 | Stratifier: raw N_eff | Denoised N_eff; raw N_eff kept as a sensitivity analysis | See the item-6 note below the table |
| 7 | Tertile cutpoints 1.707392, 2.269285 | New cutpoints on the denoised measure (Section 4.4) | The v1 value 1.707392 was rounded up: applied as written, it would place OpenML 779, the boundary dataset, in T1 (63/61/63 instead of the stated 62/62/63). Superseded by item 6 |
| 8 | Correlation table labelled "in the target region" | Corrected | The v1 values (e.g. +0.047 with auc_lr) were computed on all {{p0b_included}} Phase 0b datasets. On the {{p0b_target}} target-region datasets of the Phase 0b pool, the raw measure's correlation with auc_lr is {{p0b_raw_auclr_target}} |
| 9 | Power 0.77 at β₁ = 0.008 | Power recomputed under the registered model (Section 7) | The v1 simulation omitted the mandatory log p covariate. Re-simulated on the final-pool design ({{n_defined}} target-region datasets, {{fam_defined}} families) with the v1 regression (raw N_eff, log p, unweighted) but the v2 outcome model (family ICC 0.3), power at β₁ = 0.008 is {{pow_v1_008}} |
| 10 | Removing the strongest feature was not planned | Removal arm is **exploratory**; it was later dropped for compute before registration (Section 5.9) | Manipulation check (Section 5.5): removing the strongest feature removes {{rm1_share_top}} of the total marginal signal but does not raise the dispersion measure ({{rm1_dlog_top}} log-units, against {{rm1_dlog_rand}} for a random feature) |
| 11 | AUROC only | Adds log loss, Brier score, calibration slope and intercept | Clinical use depends on the probabilities |
| 12 | {{p0b_included}} datasets | Pool extended by the same frozen rules in Phase 0c to {{final_included}} datasets ({{n_target}} in the target region) | T1 held 17 families under v1; it holds {{T1_fam}} in the final pool. More families improve power at the narrow-signal end |
| 13 | — | The injection arm's two compared arms contain the same seven pure-noise columns, and the arm uses independent injection replicates | An independent pre-registration review showed that the first draft of the injection arm confounded concentration with the number of noise columns (Section 5.4) |
| 14 | — | Compute scope fixed before registration from a Phase 0c timing-and-variance pilot on synthetic data outside the pool; execution order and handling of shortfalls pre-specified (Section 5.9) | The second review pass found that the full design may exceed the available compute. The pilot also gives an empirical value for the SD of C_i (Section 7) |
| 15 | — | Single-threaded conventional-model fits; one CPU type for all conventional models and one GPU type for all FMs (Section 5.1) | Results of some libraries depend on the number of threads and on the machine; fixing both makes Phase 1 reproducible when it runs on several machines |
| 16 | Phase 0c reproduction tolerance 1e-6 | Tolerance based on the discrete steps of the AUROC (Section 3.3) | The 1e-6 threshold did not allow for tie reordering between machines; eleven datasets failed it for that reason only |
| 17 | Phase 0b family labels | {{n_family_corrections}} Phase 0b family labels merged into others; {{n_alias_merges}} new datasets assigned to existing families (Section 3.3) | A same-source review in Phase 0c found datasets built from the same records under different labels. Family is the unit of independence, so these would have been counted as independent evidence |

**Note on item 6: why the stratifier changed.**
- The construct check (Section 4.3) shows that raw N_eff is inflated by chance lifts of null features. This inflation largely explains its correlation of {{raw_logp}} with log p in the target region. Denoising lowers that correlation to {{dn_logp}} but raises the correlation with auc_lr to {{dn_auclr}}, hence the auc_lr covariate.
- Phase 0b's `environment.json` recorded a stratifier-selection rule: |ρ with auc_lr| ≤ 0.25, and values within [0, 2]. Under that rule Phase 0b selected N_eff / p. Draft v1 had already departed from that rule, and v2 departs further: log N_eff_dn has ρ = {{dn_auclr}} with auc_lr and a range of {{dn_range}}. A lower denoising threshold (z = 1.0) was inspected on the Phase 0b pool, where it would have met the correlation criterion (+0.23), and was not chosen; on the final pool it would not (ρ = {{dn_auclr_z1}}) (Section 4.3).
- The reason for departing is construct validity: N_eff / p does not measure how many features carry signal. N_eff / p is retained as a sensitivity stratifier (E7).

---

## 1. Background and motivation

Tabular foundation models are transformers pretrained on synthetic tables that predict by in-context learning, for example TabPFN v2 (1), TabPFN-2.5 (2), TabPFN-3 (3), TabPFN-3.5 (4) and TabICLv2 (5).

Which dataset properties decide when such models help remains an open question:
- Earlier benchmark studies asked when deep models beat gradient-boosted trees and related the answer to dataset meta-features (6, 7).
- A recent analysis of 51 TabArena datasets with false-discovery control found that global meta-features give at most weak routing signal between model families (8).
  - Its controlled experiments added nuisance feature directions or redundant copies.
  - None of them manipulated how the *predictive* signal is distributed across features at fixed total information.

We test one pre-specified mechanism. **When the predictive signal is spread over many weak features, a foundation model's prior should help more. When the signal is concentrated in a few strong features, which conventional learners exploit easily, the advantage should shrink or vanish.**

**Origin of the question.** In a controlled comparison of ten model families for CIN2+ prediction in women referred for colposcopy, including a tabular foundation model, no family outperformed logistic regression, and a clinician's image-based assessment was the most important group of variables in every family (9). That study prompted the present question. Its data are not used here: this study uses public benchmark datasets only.

---

## 2. Hypotheses

**Definitions.**
- The **advantage** of a foundation model (FM) on a dataset is Δ_sel = AUROC(FM) − AUROC(S). S is the conventional model chosen by inner cross-validation from six candidates (Section 5.3).
- The **additive advantage** is Δ_add = AUROC(FM) − AUROC(S_add). S_add is chosen by the same inner cross-validation from logistic regression and EBM only.
- The primary FM is TabPFN-3.5.

**Testing conventions.** Every confirmatory test is directional, at one-sided α = 0.025. This is equivalent to requiring the two-sided 95% confidence interval to exclude the null value in the predicted direction.

### 2.1 Primary hypotheses (fixed sequence: P1, then P2)

**P1 — intervention.**
- **Hypothesis.** Columns carrying a fixed amount of added information about the outcome are appended to each dataset. When that information is added as **one strong column plus seven pure-noise columns** (*concentrated*), the FM advantage is smaller than when it is added as **eight equally weak columns plus seven pure-noise columns** (*dispersed, noise-matched*).
- **Test quantity.** C_i = mean over four strength levels and all injection replicates (three; Section 5.9) of [Δ_sel(concentrated) − Δ_sel(dispersed-noise)].
- **Prediction.** The family-weighted mean of C_i is below 0.

**P2 — observation.**
- **Hypothesis.** Across datasets, Δ_sel increases with log N_eff_dn, adjusting for log p, log n and auc_lr.
- **Prediction.** β₁ > 0 in the regression of Section 6.3.

**Order.** P2 is tested confirmatorily only if the P1 null hypothesis is rejected. Otherwise its estimate and interval are reported and labelled exploratory. The fixed sequence controls the family-wise error rate of the primary family at one-sided 0.025.

### 2.2 Key secondary hypotheses

S1–S6 are tested only if P1's null is rejected. They are corrected by Holm at family-wise one-sided α = 0.025 within the family.

- **S1 — no practically relevant advantage where detectable marginal signal is narrowest.** In tertile T1 (narrowest), the family-weighted mean Δ_sel is below +0.01.
- **S2 — an advantage where detectable marginal signal is broadest.** In tertile T3 (broadest), the family-weighted mean Δ_sel is above 0.
- **S3 — generality, TabICLv2.** The P1 contrast with TabICLv2 as the FM is below 0.
- **S4 — generality, TabPFN-2.** The P1 contrast with TabPFN-2 as the FM is below 0.
- **S5 — sample size.** The dispersion effect is stronger at smaller training sizes: the coefficient of log N_eff_dn × log n_train is below 0 (Section 6.5).
- **S6 — specificity.** The P1 contrast computed with Δ_add in place of Δ_sel is below 0. That is, the dispersion effect holds even against the better of the two additive learners, for which the injected block is well specified.

**Error control is per family.** The family-wise error rate is controlled within the primary family and within the key secondary family, not jointly across the two. This is stated so that readers can apply a joint correction themselves.

### 2.3 Null expectation

If the FM prior confers a constant advantage, or none, regardless of how signal is distributed, then C_i has mean 0 and β₁ = 0.

---

## 3. Dataset pool

### 3.1 Frozen screening rules (unchanged from Phase 0b)

- Binary classification.
- 200 ≤ n ≤ 10,000.
- 5 ≤ p ≤ 100 after cleaning.
- Minority-class rate between 5% and 50%.
- Retained columns have ≤ 30% missing values.
- Excluded: time-series, longitudinal, and image- or text-derived high-dimensional tables.
- Identifier-like columns removed.
- Datasets with reference logistic-regression AUROC ≤ 0.52 excluded.

The exact values and regular expressions are in Phase 0b's `environment.json`.

### 3.2 Phase 0b pool

- **Screened:** 500 candidates from OpenML-CC18 (study 99), the AutoML Benchmark classification suite, and an ascending scan of the OpenML archive up to ID 44597.
- **Included:** {{p0b_included}} datasets.
- **Target region** (reference logistic-regression AUROC in [0.60, 0.85]): **{{p0b_target}} datasets in {{p0b_target_fam}} provenance families**.

### 3.3 Phase 0c extension and final pool

1. **Scan.** The ascending scan continues from ID 44598 to the highest ID available at the time of the scan, with the frozen Phase 0b code, rules and environment, and no cap on the number of candidates.
2. **Families.** New datasets receive family labels by the frozen rules. The executor listed {{alias_pairs}} candidate pairs by text similarity. Most were artefacts of shared template descriptions, so the author screened the pool by rule and reviewed the resulting pairs by hand, without access to any model result. The rules were: same normalized name, one name containing the other, same original data URL, and same n with near-equal p and class balance.
   - **Principle.** Datasets whose records come from the same underlying set of subjects or objects form one family, whatever their target variable or feature coding.
   - **Outcome.** {{n_alias_merges}} new datasets were assigned to existing families: eight versions of the German credit data to `credit-g`, and further versions of credit-approval, heart-statlog, spambase, qsar-biodeg, HELOC, bwin, the mining-accident data, aztrees, HMEQ and the COIL 2000 insurance data.
   - **Correction of Phase 0b labels.** The same review found {{n_family_corrections}} Phase 0b family labels that denote the same records as another label. These were merged: {{family_corrections_list}}. This changes family labels that Phase 0b had frozen, and it is disclosed as such (Section 0, item 17).
   - All decisions, and the pairs deliberately kept separate, are recorded in `pool_decisions.json`. Without them the target region would count {{fam_target_without_review}} families instead of {{fam_target}}.
3. **Reproduction check** (implemented in `merge_pool.classify_reproduction`). For every Phase 0b dataset, the reference logistic regression is refitted in Phase 0c, and err = |AUROC(Phase 0c) − auc_lr(Phase 0b)|.
   - err ≤ 1e-6: retained.
   - err > 1e-6: retained only if all of the following hold. Otherwise the dataset is excluded.
     - (a) every marginal AUROC of the dataset reproduces within 1e-9, so data, labels and folds are identical;
     - (b) err is a whole multiple of the tie unit 0.5 / (n1 · n0), meaning only tied or near-tied pairs changed order;
     - (c) err < 1e-3;
     - (d) target-region membership is the same at auc_lr − err and at auc_lr + err.
   - No finite err recorded: excluded.
4. **Merge.** `merge_pool.py` merges everything into the final pool.

**How the reproduction rule was set.** The Phase 0c instructions first required err ≤ 1e-6, and asked the executor to stop if more than five datasets failed. Eleven Phase 0b datasets failed, with a largest err of 1.24 × 10⁻⁴, and Phase 0c stopped.
- Every one of the eleven errors is a whole multiple of the tie unit, between 2 and 156 units.
- The Phase 0c environment has the same Python, scikit-learn, NumPy, SciPy and pandas versions as Phase 0b.
- The differences therefore arise from floating-point differences between machines and thread counts, which reorder near-tied predictions. They do not arise from a difference in the pipeline.

The 1e-6 threshold had been chosen as "effectively exact" without allowing for the discrete steps of the AUROC. The rule above replaced it before registration, without any model-comparison result. Condition (a) was then checked in Phase 0c: all eleven datasets reproduced every marginal AUROC exactly (largest error 0.0), and all eleven met conditions (b)–(d), so all were retained. The failures carry no information about any hypothesis, but the rule was changed after they were seen, and this is disclosed here.

The resumed Phase 0c run stopped once more, on its instruction to stop if an alternative stratifier was not finite: N_eff_cluster was undefined for OpenML 41538 and 44776, which have no feature above the denoising threshold. That is its value under the definition (Section 4.5). The Phase 0c instructions (`phase0c_task_brief.md`, self-check 4) allow an undefined value when its reason is stated, but the resume instructions were read as requiring a stop. The author then told the executor, in a short message that was not kept verbatim, to record the value as undefined with its reason and to continue; nothing else changed. The outputs of both stopped runs are in `phase0c_history/`.

**auc_lr used in the analyses.** Target-region membership and the auc_lr covariate use the frozen values: Phase 0b for Phase 0b datasets, Phase 0c for new ones. Phase 1 checks its logistic regression against a reference computed single-threaded on the CPU type used for the conventional models in Phase 1 (`lr_reference_phase1_preflight.csv`, from the pre-flight in Section 5.1), and requires exact agreement.

| Quantity | Phase 0b pool | Final pool |
|---|---|---|
| Included datasets | {{p0b_included}} | {{final_included}} |
| Families (all included datasets) | {{p0b_families}} | {{final_families}} |
| Target-region datasets | {{p0b_target}} | {{n_target}} |
| Target-region families | {{p0b_target_fam}} | {{fam_target}} |

The Phase 0b column uses Phase 0b's family labels; the final column uses the reviewed labels.

### 3.4 Unit of independence

Synthetic generator grids (for example `fri_c`) and versions or resamples of the same underlying data share a family label. **Family is the unit of statistical independence throughout.**
- Each dataset in an analysis set receives weight 1/m_f, where m_f is the number of datasets of its family in that analysis set. Every family therefore carries total weight 1.
- Standard errors are clustered on family.

---

## 4. Stratifier

### 4.1 Marginal AUROCs (frozen, Phase 0b)

For each feature j, `auc_j` is the out-of-fold marginal AUROC under the frozen split `StratifiedKFold(5, shuffle=True, random_state=42)`:
- the direction of association is chosen on the training fold;
- held-out values are mapped through the training fold's empirical distribution;
- the held-out values are pooled over the five folds.

These values are fixed by the hash of `marginal_aucs.json`.

### 4.2 Definitions

For a dataset with n1 positives and n0 negatives:
```
SE0      = sqrt((n0 + n1 + 1) / (12 · n0 · n1))            null standard error of an AUROC (10)
lift_j   = max(auc_j − 0.5 − 1.645 · SE0, 0)               denoised lift
N_eff_dn = (Σ_j lift_j)² / Σ_j lift_j²                     Hill number of order 2 (11)
```

**The primary stratifier is log N_eff_dn.**
- It equals 0 when a single feature carries all detectable marginal signal.
- It approaches log p when all features contribute equally.
- It measures the **breadth of detectable marginal signal**, and P2 is interpreted only in that sense (Section 6.4).

**Undefined values.** If no feature exceeds 0.5 + 1.645·SE0, N_eff_dn is undefined, and the dataset is excluded from every analysis that uses N_eff_dn. This applies to {{n_undef}} target-region dataset(s) (OpenML {{undef_ids}}). Such datasets remain in P1, which does not use N_eff_dn.

**Sensitivity measure.** The v1 measure N_eff_raw (lift_j = max(auc_j − 0.5, 0)) is retained as a sensitivity analysis.

### 4.3 Construct check on synthetic data with known dispersion

**Why a construct check.** A null feature's held-out marginal AUROC fluctuates around 0.5. Under the raw definition it therefore contributes a positive lift about half the time, so raw N_eff grows with the number of null features.

**Setup** (`construct_check.py`; results in `construct_check.csv` and `construct_check_summary.json`).
- Data come from the frozen synthetic generator (Section 5.7): k equally informative features among p = 12 or 32; Bayes AUROC 0.70 or 0.80; prevalence 0.30; n = 300, 1,000 or 3,000; four replicates per cell.
- Marginal AUROCs are computed by an emulation of the Phase 0b procedure (Section 4.1), not by the Phase 0b code itself (which is registered in `pipeline_code/`).

**Results with uncorrelated features:**

| Measure | Mean absolute error vs log k | Mean bias | Change from p = 12 to p = 32 at fixed k and n |
|---|---|---|---|
| log N_eff_raw | {{cc_raw_mae}} | {{cc_raw_bias}} | {{cc_raw_infl}} |
| **log N_eff_dn** | **{{cc_dn_mae}}** | **{{cc_dn_bias}}** | **{{cc_dn_infl}}** |
| log N_eff_cond (Section 4.5) | {{cc_cond_mae}} | {{cc_cond_bias}} | {{cc_cond_infl}} |

- **One informative feature (true log N_eff = 0).** Mean estimates at n = 300 / 1,000 / 3,000 were {{cc_k1_raw_by_n}} for the raw measure and {{cc_k1_dn_by_n}} for the denoised measure.
- **Ranking.** Within fixed p and n, the three measures rank k similarly (mean Spearman {{cc_rank}}).

**What denoising does not fix.**
1. **Dependence on n for widely dispersed signal.** When each informative feature is individually weak, some fall below the threshold at small n. At k = 32 (all features informative, p = 32), mean log N_eff_dn was {{cc_k32_dn_by_n}} at n = 300 / 1,000 / 3,000; the raw measure gave {{cc_k32_raw_by_n}}; true log k is 3.47. An additive log n covariate cannot remove this k × n dependence, so P2 is repeated with a log n × log N_eff_dn interaction and with a spline (sensitivity analyses 9–10).
2. **Redundancy.** When uninformative features are correlated with informative ones (equicorrelation 0.3), every marginal measure counts the correlates as signal sources. At k = 1, mean log N_eff_dn was {{cc_k1_dn_rho3_by_n}} (n = 300 / 1,000 / 3,000), and {{cc_rho3_k1_T3}} of those draws would fall in tertile T3.
3. **Class balance.** Prevalence was fixed at 0.30 in the check, so behaviour under other class balances was not examined.

**The conditional measure** (Section 4.5) reduced the redundancy problem at large n. However, chance coefficients of null features inflated it heavily (table above), so it failed the check with uncorrelated features. It is exploratory only (E7).

**Choice of the threshold 1.645.** The full sequence is archived in `code/exploration/` (commands and printed outputs):
1. **Synthetic comparison (step 03).** z ∈ {0, 1.0, 1.645, 1.96, 2.576} were compared on synthetic data from the frozen generator. The criteria were mean absolute error against log k and the inflation from p = 12 to p = 32. z = 1.645, the one-sided 95% null quantile, was best on mean absolute error, and its inflation was about 40% of that of z = 1.0 (0.144 against 0.353); z = 1.0 had a slightly higher rank correlation with k.
2. **Inspection on the pool (steps 04–05).** Before z was frozen, the denoised measure was also computed on the frozen Phase 0b marginal AUROCs of the target region for z ∈ {0, 1.0, 1.645, 1.96}. Its correlations with log p, log n, auc_lr and minority rate, and the resulting tertiles, were inspected. On that pool, the correlation with auc_lr was +0.231 with z = 1.0, which satisfies Phase 0b's |ρ| ≤ 0.25 rule, and +0.349 with z = 1.645. On the final pool the values are {{dn_auclr_z1}} and {{dn_auclr}}. z = 1.645 was nevertheless kept, on the synthetic criterion of step 1. No model other than the Phase 0b reference logistic regression, and no model-comparison quantity, was involved.
3. **Frozen record.** The synthetic comparison was repeated as frozen code (`threshold_choice.py`, `threshold_choice.json`) on the construct-check draws, so it is a record, not an independent validation. z = 1.645 gave mean absolute error {{tc_mae_1645}}, against {{tc_mae_1}} for z = 1.0 and {{tc_mae_196}} for z = 1.96. Its rank correlation with k was {{tc_rank_1645}} (z = 1.0: {{tc_rank_1}}), and its change from p = 12 to p = 32 was {{tc_infl_1645}} (z = 1.0: {{tc_infl_1}}).

Because the choice between z = 1.0 and z = 1.645 is a judgement between level accuracy and rank accuracy, P2 is also reported with z = 1.0 and z = 1.96 (E7).

### 4.4 Frozen cutpoints (final pool)

**Rule.** `numpy.quantile(method="linear")` at 1/3 and 2/3 of log N_eff_dn over target-region datasets with a defined value. T1: x < c1; T2: c1 ≤ x < c2; T3: x ≥ c2.

| Tertile | log N_eff_dn | N_eff_dn | Datasets | Families | Singleton families | Largest family |
|---|---|---|---|---|---|---|
{{tertile_rows}}

Spearman correlations on the {{n_defined}} target-region datasets with a defined value:

| Stratifier | log p | log n | auc_lr | Minority rate |
|---|---|---|---|---|
{{corr_rows}}

The correlation with auc_lr arises because stronger overall signal lifts more features above the noise threshold. auc_lr is therefore a mandatory covariate (Section 6.3).

### 4.5 Alternative stratifiers (computed in Phase 0c before registration; sensitivity or exploratory only)

1. **N_eff_cond (conditional).**
   - The Phase 0b reference logistic regression is refitted on each outer training fold, with identical preprocessing.
   - For each original feature j, its contribution to the linear predictor on held-out rows is c_ij = Σ over j's transformed columns of β·x̃. A numeric feature has one column; a categorical feature has its one-hot columns.
   - lift_j is the SD of c_ij over the stacked held-out rows of all five folds; N_eff_cond = (Σ lift)² / Σ lift².
   - Implementation: `phase0c_functions.neff_conditional`.
2. **N_eff_cluster.**
   - Features are clustered by average-linkage clustering on 1 − |Spearman ρ|, computed without labels, and cut at |ρ| = 0.7.
   - Cluster lift is the largest **denoised** member lift; N_eff is then computed over clusters.
   - Implementation: `phase0c_functions.neff_cluster`.
3. **Other variants:** N_eff_raw (v1), N_eff_raw / p (the Phase 0b rule's choice), N_eff_dn / p, and SCR = (auc_best − 0.5) / (auc_lr − 0.5).

None of these enters a confirmatory test. In Phase 0c, N_eff_cond was finite for all {{final_included}} datasets. N_eff_cluster was undefined for {{n_cluster_undef}} datasets (OpenML {{cluster_undef_ids}}), which have no feature above the denoising threshold (`alt_stratifiers.csv`).

---

## 5. Models, experimental arms and outcome quantities

### 5.1 Models

| Role | Model | Configuration |
|---|---|---|
| FM, primary | TabPFN-3.5 | `tabpfn` 9.0.0, `ModelVersion.V3_5` (not the Fast variant), checkpoint `tabpfn-v3.5-20260909.safetensors`, library defaults, local weights (4) |
| FM | TabICLv2 | `tabicl` 2.2.0, checkpoint `tabicl-classifier-v2-20260212.ckpt`, defaults (5) |
| FM | TabPFN-2 | `tabpfn` 9.0.0, `ModelVersion.V2`, checkpoint `tabpfn-v2-classifier-v2_default.ckpt` (the file may be loaded under its content-identical alias `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, which despite its name is not fine-tuned), defaults, local weights (1) |
| Conventional | Logistic regression | L2, C = 1, `max_iter=2000`, one-hot encoding and standardisation, identical to the Phase 0b reference pipeline |
| Conventional | EBM | `interpret` ExplainableBoostingClassifier, defaults (12) |
| Conventional | CatBoost | defaults, `verbose=0` (13) |
| Conventional | LightGBM | defaults, `verbose=-1` (14) |
| Conventional | XGBoost | defaults (15) |
| Conventional | Random forest | scikit-learn defaults (16) |
| Deep reference (main arm only) | FT-Transformer | `get_default_kwargs()`; AdamW with learning rate 1e-4 and weight decay 1e-5 on shuffled mini-batches of 256 rows; at most 100 epochs; early stopping with patience 10 on an inner stratified 80/20 split of the training fold; `BCEWithLogitsLoss` with `pos_weight` = N_neg / N_pos computed on the 80% training part (17) |

**Training data of the FM checkpoints.** TabPFN-3.5's model card and the TabICL repository state that these models were trained on synthetic data only. The TabPFN-2 model card does not state the training data. Its model repository documents `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, the file loaded in Phase 0c, as identical to the pinned `tabpfn-v2-classifier-v2_default.ckpt`, the default TabPFN-2 classifier (the SHA-256 in Section 5.1 pins the content); the statement that it was trained on synthetic data rests on the paper (1). Checkpoints fine-tuned on real data (for example Real-TabPFN-2.5) are deliberately not used. The sources and quotes are recorded in `environment_phase1.json` (`training_data_statements`), and each checkpoint's SHA-256 in Section 5.1; Phase 1 stops if a checkpoint's hash differs.

**Environment** (Phase 0c record `environment_phase1.json`; lock file `environment_phase1.lock.txt`, SHA-256 {{env_lock_sha}}). Python {{env_python}}; {{env_packages}}. GPU {{env_gpu}}, CUDA {{env_cuda}}. Checkpoint SHA-256: TabPFN-3.5 {{ckpt_tabpfn35}}; TabPFN-2 {{ckpt_tabpfn2}} (file `{{ckpt_tabpfn2_file}}`); TabICLv2 {{ckpt_tabicl2}}.

**Common settings.**
- No hyperparameter tuning.
- Random seed 13 for every model.
- FMs run on GPU with local weights and no remote API. A software sample limit that applies only on CPU is not a model limit, so such datasets are run on GPU and never subsampled.
- **Threads and machines.** Some libraries give different results with different numbers of threads; the LightGBM documentation, for example, states this for its default setting. Every fit of a conventional model, including every inner-selection fit, therefore uses a single thread (`n_jobs`, `thread_count` or `num_threads` = 1; BLAS and OpenMP limited to one thread), and parallelism comes only from running many fits at once. These are execution settings, not hyperparameters. All conventional-model fits run on one CPU type, and all FM and FT-Transformer fits on one GPU type, with the registered environment on every machine; no model is split across machine types. The CPU and GPU models are recorded for every fit. Hardware: conventional models on {{pf_hw}}; FMs and FT-Transformer on NVIDIA GeForce RTX 4090. Each dataset is assigned to one CPU instance, which runs all of that dataset's conventional-model fits in every module, so no within-dataset contrast crosses machines. An instance other than the pre-flight instance, including the second planned one, is used only after it reproduces the pre-flight logistic-regression reference on every pool dataset to within 1e-12. The CPU model and host are recorded for every fit. A pre-flight on the first instance, run before registration (`phase1_preflight_brief.md`; outputs in `preflight_outputs/`), recomputed the logistic-regression reference there and compared it with the Phase 0c reference under conditions (b) and (c) of Section 3.3: {{pf_lr}}. It also timed the conventional models on the synthetic pilot datasets; the median ratio of its fit times to the pilot machine's was {{pf_ebm}} for EBM and {{pf_all}} over all models. It delivered the executor's code (`pipeline_code/`: the Phase 0b and Phase 0c code, unchanged), which is registered here with the SHA-256 of each file (`pipeline_code_sha256.txt`). Phase 1 imports the conventional models' preprocessing and fitting unchanged from this code (the Phase 0b logistic-regression pipeline for LR; the pilot's fit function for the other five). The pilot's FM and FT-Transformer functions were written for numeric-only synthetic data: they do not declare categorical columns or impute missing values as Section 5.2 requires, and FT-Transformer is trained full-batch. Phase 1 therefore does not use them. The Phase 1 driver (data loading, variants from the frozen generators, fold loops, inner selection as in Section 5.3, FM and FT-Transformer input preparation and training as in Sections 5.1–5.2, output writing) is written after registration. It is smoke-tested on a toy dataset with categorical columns and missing values, and committed with its SHA-256 to the public repository before the first model fit on any pool dataset. It may change afterwards only to fix a reported error, which is recorded as a deviation.

### 5.2 Common protocol

- **Outer split.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, identical to Phase 0b and shared by every model and arm of a dataset.
- **Preprocessing.** Fitted within each training fold and applied to the held-out fold.
  - Imputation: median for numeric columns, mode for categorical columns, identical for all models. FMs and EBM do not use their own missing-value handling.
  - LR and FT-Transformer: one-hot encoding with `handle_unknown='ignore'`, plus standardisation.
  - Tree models, EBM and FMs: integer encoding with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`. Categorical columns are declared through the library's categorical-feature argument where one exists, as fixed after the Phase 0c smoke test: `categorical_features_indices` (TabPFN), pandas `category` dtype (TabICLv2), `feature_types="nominal"` (EBM), `cat_features` with string-coded categories (CatBoost), `categorical_feature` (LightGBM), `enable_categorical=True` with pandas `category` dtype (XGBoost). Random forest has no categorical argument and uses the integer codes.
- **Probabilities** are the raw, uncalibrated `predict_proba` output.

### 5.3 The selected comparators S and S_add

- **Inner split.** Within each outer training fold, `StratifiedKFold(5, shuffle=True, random_state=13)` defines inner folds.
- **Inner fits.** Each of the six conventional models is fitted on each inner training split, with preprocessing refitted inside that split. Its mean inner held-out AUROC is recorded.
- **Selection of S.** The model with the highest mean is selected. Ties are broken by the order LR, EBM, CatBoost, LightGBM, XGBoost, random forest.
- **Selection of S_add.** S_add is chosen the same way from LR and EBM only.
- **Outer predictions.** The selected model's predictions on the outer held-out fold are those of the same model fitted on the whole outer training fold.
- **Failures.** A candidate that fails in any inner fold is excluded from selection in that outer fold, and the exclusion is recorded. If every candidate fails, S (or S_add) is undefined for that fold.
- **No leakage.** Selection uses no outer held-out data.

### 5.4 Injection arm (P1, S3, S4, S6) — frozen in `generators.injection_block`

**Datasets:** all target-region datasets.

**Construction.** For each dataset and replicate r (r ∈ {0, 1, 2}; Section 5.9), columns are appended to the cleaned feature matrix once, before the outer split. Let d = √2·Φ⁻¹(ℓ), ε ~ N(0, 1), and level ℓ ∈ {0.65, 0.75, 0.85, 0.95}.

| Arm | Columns | Content |
|---|---|---|
| noise | 8 | 8 pure-noise columns (reference) |
| concentrated, ℓ | 8 | one column z = d·y + ε (marginal AUROC ℓ) and the 7 pure-noise columns of the dispersed-noise arm |
| dispersed, ℓ | 8 | 8 columns z = (d/√8)·y + ε (column-matched; exploratory, E10) |
| **dispersed-noise, ℓ** | 15 | the 8 dispersed columns and 7 further pure-noise columns, in a seeded order (**noise-matched; used in P1**) |

This gives 13 variants per replicate, and 39 per dataset with three replicates.

**Information matching.**
- Signal-bearing columns are conditionally independent given y. The Bayes AUROC of the injected block is therefore Φ(d/√2) = ℓ in every signal arm; this is verified in `test_generators.py`.
- *Concentrated* and *dispersed-noise* contain the same seven pure-noise columns. They differ in how the information is spread (1 strong column versus 8 weak ones) and therefore in total column count (8 versus 15).
- The difference in column count is part of the manipulation: the seven additional columns are the weak signal-bearing columns themselves, and it cannot be removed while holding the information and the pure-noise columns fixed. Its effect on the contrast can go either way. If extra columns hurt the FM more than S (for example if the FM is sensitive to additional columns), the P1 contrast is biased **against** the hypothesis. If they hurt S more than the FM (for example through column subsampling in tree ensembles), it is biased **towards** the hypothesis. The column-matched contrast and the noise arm versus the original data (E10) show how large column-count effects are.

**Common random numbers.** Within a dataset and replicate, all arms and levels are built from the same draws. The concentrated arm's seven pure-noise columns are exactly those of the dispersed-noise arm, and the noise of its strong column is the noise of the weak column in the same position. The position of the concentrated column and the order of the 15 columns are shared across levels.

**Models.** All models except FT-Transformer, with the full protocol of Sections 5.2–5.3.

**Manipulation check** (`manipulation_check.csv`, `manipulation_summary.json`). This uses the population marginal AUROCs of the injected columns. Family-weighted mean change in log N_eff_dn:

| ℓ | Concentrated | Dispersed and dispersed-noise | Datasets where a dispersed column falls below the denoising threshold |
|---|---|---|---|
{{manip_rows}}

For comparison, the distance between the two tertile cutpoints is {{cut_gap}} log-units. The manipulation is weakest at ℓ = 0.65.

### 5.5 Removal arm (exploratory, E1) — frozen in `removal_plan.json`; not run

This arm is not run (scope decision, Section 5.9). Its design and manipulation check are kept because the manipulation check is cited in Sections 0 and 5.9.

**Construction.**
- The top-1 and top-3 features by frozen marginal AUROC are removed; ties are broken by name.
- The control removes the same number of features drawn at random from the rest, with five draws each, seeded by dataset ID.
- The top-3 arm requires p ≥ 6: {{n_top3}} datasets.
- Models as in 5.4.

**Why this arm is exploratory: manipulation check.**

| Removal | Share of total marginal lift removed | Change in log N_eff_dn |
|---|---|---|
{{removal_rows}}

Removing the strongest features mainly removes signal; it does not make the remaining signal more dispersed.

The arm was planned because it answers a practical question: what happens to the FM advantage when the strongest predictor is unavailable. It is not a test of the dispersion hypothesis.

### 5.6 Sample-size arm (S5)

- **Datasets:** all target-region datasets.
- **Subsampling.** In each outer fold, the training fold is subsampled (stratified, `generators.subsample_indices`) to n_train ∈ {100, 250, 500, 1000}, with five repetitions per size. Each subsample is evaluated on the full held-out fold.
- **Eligibility** (`generators.subsample_eligible`). A size is run only if it is smaller than the smallest outer training fold and the expected number of positives is at least 10.
- **Full-size level.** The main arm supplies the full-training-fold level. For it, n_train is the mean size of the five outer training folds.
- **Models.** All models except FT-Transformer. Inner selection (Section 5.3) runs within each subsample.

### 5.7 Synthetic mechanism arm (exploratory, E2) — frozen in `generators.synth_draw` and `synth_calibration.json`

**Features.** p = 32 equicorrelated N(0, 1) features, with correlation ρ ∈ {0, 0.3}. Of these, k ∈ {1, 2, 4, 8, 16, 32} are informative with equal weight.

**Outcome.** P(y = 1 | x) = logistic(b0 + s·Σ_{j≤k} g(x_j)/√k), where:
- g(x) = x (linear generator) or g(x) = sign(x) (threshold generator);
- s and b0 are calibrated exactly by quadrature, so that the Bayes AUROC is 0.70 or 0.80 and the prevalence is 0.30.

**Design.**
- n_train ∈ {100, 300, 1000, 3000}.
- One test set of 5,000 per cell and replicate, shared by all training sizes. The column permutation is also shared across training sizes and the test set.
- 20 replicates; 192 cells.
- All FMs and the six conventional models are fitted, with inner selection on the training set.

### 5.8 Outcome quantities

For model m on a dataset, AUROC_m is the **mean of the five per-fold held-out AUROCs**. For S and S_add, fold k uses the model selected in fold k.

| Symbol | Definition | Role |
|---|---|---|
| **Δ_sel** | AUROC(FM) − AUROC(S) | Primary outcome |
| Δ_add | AUROC(FM) − AUROC(S_add) | S6 |
| **C_i** | Mean over ℓ ∈ {0.65, 0.75, 0.85, 0.95} and all injection replicates r of [Δ_sel(concentrated, ℓ, r) − Δ_sel(dispersed-noise, ℓ, r)] | P1, S3, S4 |
| C_i^add | As C_i, with Δ_add | S6 |
| C_i^col | As C_i, with the column-matched dispersed arm | Exploratory (E10) |
| Δ_CB, Δ_LR, Δ_EBM, Δ_FTT | AUROC(FM) − AUROC(fixed comparator) | Exploratory (E5); Δ_CB is the v1 primary outcome |
| Δ_oracle | AUROC(FM) − max over the six conventional AUROCs | Exploratory (E5); biased against the FM by maximum selection |
| ΔLogLoss, ΔBrier | FM − S, on pooled out-of-fold raw probabilities | Exploratory (E4) |
| Calibration slope, intercept | Logistic recalibration of y on logit(p) over the pooled out-of-fold predictions, with p clipped to [1e−6, 1 − 1e−6] | Exploratory (E4) |

In the synthetic arm, AUROC is computed on the 5,000-row test set.

### 5.9 Modules, compute scope and execution order

Phase 1 runs six modules:

| Module | Content | Used for |
|---|---|---|
| M1 | Main arm: every model on every final-pool dataset | P2, S1, S2, E3–E9, E11 |
| M2 | Injection arm (Section 5.4) | P1, S3, S4, S6, E8 (P1 part), E10 |
| M3 | Removal arm (Section 5.5): **not run** (Section 5.9) | (E1) |
| M4 | Sample-size arm (Section 5.6) | S5, E12 |
| M5 | Synthetic arm (Section 5.7) | E2 |
| M6 | Marginal AUROCs within each outer training fold (no model fitted) | E3 |

**Pilot in Phase 0c (before registration).**
- The complete Phase 1 pipeline of M2 (all arms and levels, replicate 0, all models with inner selection) is run on 12 synthetic datasets from `sklearn.datasets.make_classification` that are not in the pool: n ∈ {500, 1000, 2000} × p ∈ {10, 30} × informative share ∈ {0.3, 0.8}. The exact call is in the Phase 0c brief.
- It yields per-fit run times for every model, a projected run time for every module on the final pool, and the SD of C_i across the 12 pilot datasets.
- The pilot may change only two things: the power statement (Section 7) and the compute scope below. Hypotheses, MESOI, outcome definitions, analyses and tests are fixed before it is run.

**Scope rule (fixed before the pilot).** If the projected run time exceeds the compute available, the scope is reduced, from run times only, in this order:
1. M5;
2. M3;
3. the noise and column-matched dispersed arms of M2 (E10);
4. the third injection replicate;
5. the second injection replicate.

M1, M4, M6, and the concentrated and dispersed-noise arms of M2 with replicate 0 are never dropped.

**Pilot, pre-flight and decision.**
- The pilot completed in Phase 0c. Its projection (`compute_projection.md`) is {{cpu_core_hours}} single-thread CPU core-hours for the conventional models, {{ebm_share}} of them for EBM, and {{gpu_hours}} GPU hours.
- The pre-flight found the Phase 1 CPU type slower per fit than the pilot machine (median EBM time ratio {{pf_ebm}}; Section 5.1). On one instance, all modules would have taken {{pf_wall_one_all}}.
- The compute available is {{compute_limit}} on {{n_instances_word}} instances of this CPU type. The author set this limit when taking the decision, after the pre-flight timings were known, from the study schedule. On them, all modules would take {{w_all}}; dropping M5 alone, the first step of the order above, {{w_nom5}}; dropping M3 alone {{w_nom3}}.
- **Decision (before registration, from run times only): {{scope_decision}}** It departs from the order above: the order would have dropped both M5 and M3, whereas only M3 is dropped. M5 is one of the study's three lines of evidence and takes about {{m5_share}} of the conventional-model CPU work; M3 takes about {{m3_share}}, and its manipulation check (Section 5.5) shows that removing the strongest feature does not raise the dispersion measure, so M3 would add little. No model-comparison result on pool data existed when the decision was made.
- With this decision, the projected run time is {{pf_wall}}.

**Execution order after registration:** M1, M6, M2 (concentrated and dispersed-noise arms: replicate 0, then 1, then 2), M4, M2 (remaining arms), M5. M5 runs on the pre-flight instance. If compute nonetheless runs out, the completed modules are analysed as registered, and the shortfall is reported as a deviation (Section 11).

---

## 6. Analysis plan

### 6.1 Estimator (registered as code: `stats_core.py`)

- Family-weighted least squares, with weights 1/m_f computed within the analysis set actually used.
- CR1 cluster-robust covariance, clustered on family.
- A t reference distribution with G − 1 degrees of freedom, where G is the number of families in the analysis set.
- One-sided p-values from `stats_core.one_sided_p`.
- A family-weighted mean is the intercept-only case.

### 6.2 P1 (primary, tested first)

- **Analysis set.** Target-region datasets for which TabPFN-3.5 and all six conventional models completed all five folds in every concentrated and dispersed-noise variant, with S defined in every fold.
- **Test.** H0: mean C ≥ 0 against H1: mean C < 0, using the one-sided p of the family-weighted mean.
- **Minimum effect size of interest (MESOI):** −0.005.

### 6.3 P2 (primary, tested only if the P1 null is rejected)

```
Δ_sel ~ β0 + β1·log N_eff_dn + β2·log p + β3·log n + β4·auc_lr
```

- **Analysis set.** Target-region datasets with a defined N_eff_dn for which TabPFN-3.5 and all six conventional models completed all five folds of the main arm. An FT-Transformer failure does not exclude a dataset.
- **Covariates.** All four are mandatory and are kept regardless of significance.
- **Test.** H0: β1 ≤ 0 against H1: β1 > 0.
- **MESOI:** β1 = 0.008 AUROC per log-unit (unchanged from v1).

### 6.4 Decisions and interpretation

**Statistical decision** (used for the fixed sequence and for gatekeeping):
- rejection at one-sided 0.025 for P1 and P2;
- rejection at the Holm-adjusted level for S1–S6.

**Substantive conclusion** for P1 and P2, from the two-sided 95% confidence interval (equivalent to the one-sided test at 0.025). "Reaches the MESOI" means the interval contains values at least as large as the MESOI in the predicted direction (P1: lower bound ≤ −0.005; P2: upper bound ≥ 0.008).

| Outcome | Conclusion |
|---|---|
| H0 rejected, and the CI reaches the MESOI | **Supported** |
| H0 rejected, but the whole CI lies between 0 and the MESOI | **Detectable but smaller than the MESOI** |
| H0 not rejected, and the CI reaches the MESOI | **Inconclusive** |
| H0 not rejected, and the CI does not reach the MESOI | **Evidence against a practically relevant effect** |
| The whole CI lies on the opposite side of 0 | **Evidence of an effect in the opposite direction** (this row takes precedence over the previous one) |

In the reading table below, "Supported" and "Detectable but small" are the first two rows; "Not supported" means any of the last three. The specific row is always reported.

**Reading of the primary results:**

| P1 | P2 | Reading |
|---|---|---|
| Detectable but small | Any | The dispersion effect exists under control but is below the size that matters for model choice. Because P1's null was rejected, P2 and S1–S6 are tested confirmatorily and are read with this qualification |
| Supported | Supported | Signal dispersion moderates the FM advantage under control, and the breadth of detectable marginal signal explains natural variation between datasets |
| Supported | Detectable but small | Signal dispersion moderates the FM advantage under control; measured breadth of marginal signal is associated with natural variation between datasets, but by less than the size that matters for model choice |
| Supported | Not supported | The mechanism operates when dispersion is manipulated, but measured breadth of marginal signal does not explain between-dataset differences, because of measurement limits (Section 4.3) or other dominant factors |
| Not supported | Estimate > 0 | Exploratory association only; it cannot be read as evidence for the mechanism |
| Not supported | Other | No support for the hypothesis |

**S6 qualifies P1.** If P1 is supported but S6 is not, the effect is read as one shared with smooth additive learners, not as specific to foundation-model priors.

### 6.5 Key secondary analyses

Tested only if the P1 null is rejected. Holm correction is applied across the six one-sided p-values, at family-wise α = 0.025.

- **S1.** Family-weighted mean Δ_sel in T1, with weights 1/m_f within T1.
  - Test: H0: μ ≥ +0.01 against H1: μ < +0.01.
  - The upper confidence bound is reported at the Holm-adjusted level.
- **S2.** The same in T3. Test: H0: μ ≤ 0.
- **S3, S4.** P1 with TabICLv2 and with TabPFN-2 as the FM, on the analysis set of 6.2 defined for that FM. Test: H0: mean C ≥ 0.
- **S5.** For each dataset i and size level s (the subsample sizes plus the full training fold), Δ_sel(i, s) is averaged over the five repetitions. Then:
  ```
  Δ_sel(i,s) ~ γ0 + γ1·c_neff + γ2·c_n + γ3·(c_neff × c_n) + γ4·log p + γ5·auc_lr
  ```
  - c_neff = log N_eff_dn minus its mean over the analysis set; c_n = log n_train − log 500.
  - Weights: `stats_core.hierarchical_weights`, computed on the rows analysed. A row of dataset i in family f receives 1/(m_f · L_i), where L_i is the number of that dataset's size levels in the analysis, so every family carries total weight 1 and every dataset within a family equal weight. Clustered on family.
  - Analysis set: dataset-size rows for which TabPFN-3.5 and all six conventional models completed every fold and repetition, with S defined in every fold.
  - Test: H0: γ3 ≥ 0.
- **S6.** P1 with C_i^add, on the analysis set of 6.2 restricted to datasets in which S_add is also defined in every fold of every variant used. Test: H0: mean C^add ≥ 0.

### 6.6 Exploratory analyses (pre-specified, not confirmatory)

| Code | Analysis |
|---|---|
| E1 | **Not run (Section 5.9).** Removal arm: family-weighted mean of Δ_sel(top-k removed) − mean over draws of Δ_sel(random-k removed), for k = 1 and 3; the same contrast by tertile; the change in the AUROC of S |
| E2 | Synthetic arm: mean Δ per cell with 95% intervals over replicates. Within each generator, Δ ~ log k × log n_train + ρ + Bayes AUROC. Expectations stated in advance: (a) linear generator, ρ = 0: FM − CatBoost increases with k, while FM − LR shows no trend; (b) the increase is larger at smaller n_train; (c) at ρ = 0.3, marginal dispersion measures misclassify concentrated cells as dispersed |
| E3 | Routing (see the E3 details below the table) |
| E4 | ΔLogLoss, ΔBrier, calibration slope and intercept: the P2 regression with each as outcome, and summaries by tertile |
| E5 | The P2 regression with Δ_CB, Δ_LR, Δ_EBM, Δ_FTT and Δ_oracle as outcomes |
| E6 | Dose–response shape: a restricted cubic spline in log N_eff_dn (knots at the 10th, 50th and 90th percentiles) |
| E7 | The P2 regression with N_eff_cond, N_eff_cluster, N_eff_raw, N_eff_raw / p, N_eff_dn / p, SCR, and N_eff_dn with z = 1.0 and with z = 1.96, as stratifiers |
| E8 | Medical subset: P1 and P2 within datasets labelled clinical or biomedical (`medical_labels.csv`; {{med_target}} target-region datasets in {{med_target_fam}} families) |
| E9 | The P2 regression on all included datasets, not only the target region |
| E10 | Injection arm: contrasts by level and by replicate; the column-matched contrast C^col; noise arm versus original data (the effect of adding columns alone); P1 with each fixed comparator |
| E11 | The P2 regression with TabICLv2 and TabPFN-2 as the FM |
| E12 | Main effect of n_train on Δ_sel in the sample-size arm |

**E3 details.**
- *Rule R_fixed:* in an outer fold, use the FM if log N_eff_dn computed on that training fold (nested; Phase 1 module M6) is ≥ c1; otherwise use S.
- *Rule R_lofo:* the threshold is learned leave-one-family-out.
- *Comparators:* always use the FM; always use S; inner-CV selection among all nine models (FM inner-CV results come from module M1); and a meta-feature router (leave-one-family-out logistic regression on log n, log p, minority rate, categorical share and missing share).
- *Metric:* family-weighted mean regret, where regret = max(AUROC_FM, AUROC_S) − AUROC(chosen).

### 6.7 Sensitivity analyses (applied to P1 and P2)

1. Unweighted least squares with family-clustered SEs.
2. A random-intercept mixed model by family (REML), with cluster-robust SEs.
3. One dataset drawn at random per family, repeated 2,000 times; the distribution of the estimate is reported.
4. Each of the three largest families excluded in turn.
5. Target region widened to [0.55, 0.90] and narrowed to [0.65, 0.80].
6. P2 without the auc_lr covariate.
7. Pooled out-of-fold AUROC instead of the fold mean (fixed comparators only).
8. P2 with the raw v1 stratifier and the v1 covariate set.
9. P2 with an added log n × log N_eff_dn interaction (Section 4.3).
10. P2 with a restricted cubic spline in log N_eff_dn plus the four covariates.
11. P1 by replicate, and P1 excluding level 0.65.

### 6.8 Failed runs

- A model that fails on a dataset variant after one retry (on GPU, for the FMs) is recorded, with the reason, and the variant is excluded from every analysis that needs that model.
- Datasets are never subsampled to avoid a failure.
- If more than 10% of the analysis set is lost for any primary or key secondary test:
  - the losses are described by tertile, n and p;
  - the analysis is repeated with the affected families removed entirely.

---

## 7. Power (simulated outcomes on the real design; `power_sim.py`, `power_results.json`)

**Model.** Outcome = effect + family effect + residual. The family intraclass correlation is 0.3, with 2,000 replicates (1,000 for S5, whose variance split is given in `power_sim.py`), and each test uses the registered estimator. "SD" is the total SD of the outcome across datasets.

**The between-dataset SD of C_i is unknown.** No injection run has been made on any pool dataset. Averaging over four levels and the replicates reduces the estimation-noise part of that SD, but not the true heterogeneity. The tables therefore span SD 0.01–0.03 (0.015–0.03 for P2).

**Pilot value.** Across the 12 synthetic pilot datasets of Section 5.9, with one injection replicate, the SD of C_i was {{pilot_sd_tabpfn35}} for TabPFN-3.5, {{pilot_sd_tabicl2}} for TabICLv2 and {{pilot_sd_tabpfn2}} for TabPFN-2. This is an indication only. The pilot datasets are synthetic and few, and real datasets are likely to be more heterogeneous. Three replicates reduce the estimation-noise part.

**P2.** Slope on log N_eff_dn with covariates log p, log n and auc_lr; {{n_defined}} datasets in {{fam_defined}} families:

| β1 | SD 0.015 | SD 0.020 | SD 0.030 |
|---|---|---|---|
{{pow_p2_rows}}

**P1, S3, S4 and S6.** Family-weighted mean contrast below 0, simulated on the {{n_defined}} datasets with a defined N_eff_dn ({{fam_defined}} families; P1 itself uses all {{n_target}} target-region datasets):

| True mean C | SD 0.01 | SD 0.02 | SD 0.03 | S3/S4/S6 at the worst Holm threshold, SD 0.02 |
|---|---|---|---|---|
{{pow_p1_rows}}

**Other key secondary tests:**
- **S1** (T1, {{T1_fam}} families, true mean 0): {{pow_s1}}.
- **S2** (T3, {{T3_fam}} families, true mean 0.01), unadjusted / at the worst Holm threshold: {{pow_s2}}.
- **S5** (interaction −0.005), unadjusted / at the worst Holm threshold: {{pow_s5}}.

The worst Holm threshold is one-sided 0.025/6.

**Reading.**
- P1 is well powered for contrasts of magnitude ≥ 0.004 if the SD of C_i is at most about 0.01 (power {{p1_sd01_004}}). At SD 0.02, its power at the MESOI is {{p1_sd02_mesoi}}.
- At SD 0.02, P2 has power {{p2_sd02_mesoi}} at its MESOI and reaches 0.8 at β1 = {{p2_beta80}}.
- S1 is limited by the small number of families in T1. A non-significant S1 is not evidence of an advantage.

---

## 8. Falsification

The outcome of P1 and of P2 is classified by the table in Section 6.4. In those terms:

- **Falsified.** The hypothesis is falsified as a practically relevant effect if the outcome is "evidence against a practically relevant effect" (the null is not rejected and the 95% CI does not reach the MESOI: for P1, lower bound > −0.005; for P2, upper bound < 0.008), or "evidence of an effect in the opposite direction".
- **Inconclusive, not falsified.** The null is not rejected but the CI still reaches the MESOI. The study then neither supports nor rules out the hypothesis.
- **Detectable but smaller than the MESOI.** The null is rejected but the whole CI lies between 0 and the MESOI. The effect exists but is ruled out as practically relevant.

A null, inconclusive or negative result is the study's finding and will be submitted for publication regardless of its direction.

---

## 9. Limitations declared in advance

1. **The injected columns are idealised.** They are Gaussian, linear in their effect and conditionally independent of the original features given y. For such columns, logistic regression is correctly specified and trees are at their weakest; S6 addresses this. Real strong predictors, such as a clinician's judgement, correlate with other features. P1 isolates the distribution of added information; it does not reproduce a clinical variable.
2. **N_eff_dn measures the breadth of detectable marginal signal, not dispersion itself.** It counts redundant correlates as signal sources and depends on n when the signal is widely dispersed (Section 4.3). No measure we examined removed both problems. P1 is unaffected, because its injected columns are uncorrelated with each other and with the data given y.
3. **N_eff_dn correlates with auc_lr ({{dn_auclr}}).** This is adjusted for; residual confounding with overall predictability remains possible. Sensitivity analysis 6 reports the unadjusted estimate.
4. **Few families at the narrow-signal end.** T1 contains {{T1_fam}} families, and its largest family is {{T1_largest}}. Family weighting protects inference but not power.
5. **Public benchmarks are not a random sample of clinical tables.** The medical subset is small and exploratory.
6. **Foundation models change quickly.** Versions are pinned, and conclusions apply to those versions. Using three models from two developers reduces dependence on one model but does not remove it.
7. **Indirect benchmark exposure.** The checkpoints were trained on synthetic data only, but their developers select designs on public benchmarks (for example TabArena, which draws on OpenML). This may raise their level on familiar datasets. It affects between-dataset comparisons (P2) more than the within-dataset contrast (P1).
8. **Uncertainty is conditional on the fitted models and one outer fold assignment.** Refitting variability is not included.
9. **Licensing.**
   - TabPFN-3.5 weights: licensed for non-commercial use.
   - TabPFN-2 weights: Prior Labs License.
   - TabICL: BSD-3-Clause.
   - This study is non-commercial research and redistributes no weights.
10. **Identifier-like columns.** The frozen screening rule removes identifier-like columns by name (Section 3.1) and does not catch every such column. OpenML 481 (`biomed`) keeps `Hospital_identification_number_for_blood_sample`, with marginal AUROC 0.781. The dataset lies outside the target region, so it enters no confirmatory test (only M1 and E9). The frozen rule is applied as written, and no dataset is edited by hand.

---

## 10. Declaration of prior observation

**Has been observed:**
- Outside the pool: the results reported in (9), the author's earlier applied work that prompted the question (Section 1), and an exploratory analysis of that cohort made before this study, in which removing the strongest variable was followed by an advantage of a foundation model. One sentence of `phase0_task_brief.md` that summarised that analysis has been removed (the place is marked in the file), because the cohort's data and results are not part of this study. Those data are not used in this study.
- For every pool dataset (Phase 0b):
  - the per-feature marginal AUROCs;
  - the reference logistic-regression AUROC;
  - the distributions of candidate stratifiers and their correlations with n, p, minority rate and auc_lr.
- In Phase 0c, for every pool dataset:
  - OpenML metadata;
  - pairwise feature correlations, computed without labels;
  - reference logistic-regression coefficients by fold;
  - the reproduced reference logistic-regression AUROC and its difference from the frozen value (Section 3.3);
  - categorical and missing-value shares;
  - the executor's alias candidate list, which the author screened to reach `pool_decisions.json`.
- Synthetic data only (no pool dataset):
  - the calibration of the generator;
  - the construct check and the threshold comparison, which fit only logistic regression;
  - a toy run by the independent pre-registration reviewer, on `make_classification` data, of LR, random forest and histogram gradient boosting with injected columns. It motivated item 13 of Section 0.
- The informal explorations archived in `code/exploration/`: synthetic data, and the denoised measure computed on the frozen pool marginal AUROCs for z ∈ {0, 1.0, 1.645, 1.96}, with its correlations and tertiles, before z was frozen (Section 4.3).
- In Phase 0c, on 12 synthetic `make_classification` datasets outside the pool: the results of the injection-arm pilot (Section 5.9), including AUROCs, Δ values and C_i. They were delivered in full. The author looked at the between-dataset SD of C_i (Section 7) and, inadvertently, at the two C_i values of one pilot dataset, but did not compute or inspect their means.
- In the pre-flight (Section 5.1), on the Phase 1 CPU type: the reference logistic-regression AUROC of every pool dataset, recomputed, and its difference from the Phase 0c reference; and fit times of the conventional models on the 12 synthetic pilot datasets (no performance metric).
- Manipulation checks computed from frozen marginal AUROCs.
- Power simulations with simulated outcomes.
- A smoke test of each foundation model on a `make_classification` toy dataset, to verify offline inference. No performance metric was computed.

**Has not been observed, by design:**
- **No fit of any foundation model, gradient-boosted tree, random forest, EBM or FT-Transformer on any pool dataset.**
- **No value of any Δ, C_i or regret on any pool dataset.**
- No relationship between any stratifier and any model-comparison quantity.
- Published per-dataset results for these datasets (for example OpenML run records or benchmark leaderboards) have not been consulted by the author, and will not be before Phase 1 delivery.

**Supporting evidence.**
- Phases 0, 0b and 0c and the pre-flight were run under written instructions prohibiting model comparison on pool data.
- Each delivery reported a search of the executor's code showing `LogisticRegression` as the only model fitted on pool data; that code is registered (`pipeline_code/`).
- Those instructions are registered (Section 12).

---

## 11. Deviations

Any departure from this plan will be recorded in a dated deviation log released with the manuscript, stating what changed, when and why. Both the registered and the revised analyses will be reported.

The following will not change under any circumstances:
- the final pool and its family labels;
- the stratifier definitions and cutpoints;
- the injection construction, including levels, arms and the number of replicates fixed at registration;
- the primary outcome and contrast definitions;
- the primary regression specification;
- the testing order and the multiplicity procedure.

A shortfall of compute after registration is handled as in Section 5.9 and reported as a deviation.

---

## 12. Frozen artefacts and time-stamping

All files below are committed together to a public Git repository. `MANIFEST.sha256` in that commit records the SHA-256 of each file. The commit identifier verifies every byte, and the hosting service's push record time-stamps it. The same file set is deposited on OSF.

| File | Content |
|---|---|
| `preregistration_v2.md` | This plan |
| `datasets_stratifiers.csv`, `marginal_aucs.json`, `environment.json` | Phase 0b artefacts (hashes as in draft v1: 5BB2A3CE…, 0C6D2370…, CCAC64AE…) |
| `phase0c_outputs/` (21 files) | Extended pool, reproduction checks, alias candidates, metadata, alternative stratifiers, environment and lock file, smoke test, pilot timing and results, compute projection, logistic-regression reference |
| `cleaned_data.zip` (not committed; SHA-256 `93d97f647acc7ebbdde1e55afc95c5cd55eb5cb9fd814f3070356676b724c646`) | Cleaned data of every dataset with the outer-fold column, written by the Phase 0b cleaning code in Phase 0c. Not committed because of size; re-derivable from OpenML with that code. Phase 1 reads only this archive and checks its hash first |
| `preflight_outputs/`, `pipeline_code/`, `pipeline_code_sha256.txt`, `compute_plan.json` | Pre-flight outputs (logistic-regression reference and timing on the Phase 1 CPU type, CPU record, report), the executor's code with the SHA-256 of each file, and the CPU instances and scope decision (Sections 5.1 and 5.9) |
| `pool_decisions.json`, `medical_labels.csv`, `final_pool/` | Family decisions, medical-subset labels, and the merged final pool (`datasets_stratifiers.csv`, `marginal_aucs.json`, `merge_report.json`) |
| `generator_fingerprints_phase1env.json` | Fingerprints of the frozen generators computed in the Phase 1 environment; Phase 1 must reproduce them exactly |
| `stratifier_v2.csv`, `cutpoints_v2.json` | Stratifier values and cutpoints of the final pool |
| `removal_plan.json`, `manipulation_check.csv`, `manipulation_summary.json` | Removal sets and manipulation checks |
| `synth_calibration.json`, `construct_check.csv`, `construct_check_summary.json`, `threshold_choice.json` | Synthetic calibration, construct check, threshold comparison |
| `code/exploration/` | Informal explorations that preceded the frozen stratifier (record only) |
| `README.md`, `preregistration_v2.template.md`, `fill_prereg.py`, `frozen/generator_fingerprints.json` | Repository guide; the plan with placeholders and the script that fills it; generator fingerprints in the development environment (they differ from the Phase 1 file only in the recorded NumPy version) |
| `phase0b_outputs/stratifier_report.md`, `run_log.txt`; `phase0c_history/`; `pool_review/alias_shortlist_raw.csv`; `construct_check_log.txt`, `threshold_choice_log.txt`, `power_log.txt` | Records only: the Phase 0b report and log, the outputs of the two stopped Phase 0c runs (Section 3.3), the same-size alias screen behind the family review, and run logs |
| `power_results.json` | Power simulations |
| `generators.py`, `stratifier.py`, `phase0c_functions.py`, `stats_core.py`, `merge_pool.py`, their tests, and the build, calibration, check, power and fingerprint scripts | Frozen code |
| `phase0_task_brief.md`, `phase0b_task_brief.md`, `phase0c_task_brief.md`, `phase0c_resume_brief.md`, `phase1_preflight_brief.md`, `phase1_task_brief_v2.md` | Instructions to the executor, including the prohibition on model comparison |

The repository address, the commit SHA, the push time (UTC) and the SHA-256 of `MANIFEST.sha256` are entered in the OSF registration form, because a committed file cannot contain its own commit identifier. The OSF registration DOI is cited in the manuscript.

**Later commits.** Before the first model fit on any pool dataset, the Phase 1 driver, its SHA-256 and the smoke-test log are committed to the same repository (Section 5.1). That commit and its time stamp are cited in the manuscript.

**Analysis code.** The estimators are registered as code (`stats_core.py`). The full analysis script will call only these functions and the definitions above. It will be written after Phase 1 delivery and released with the manuscript.

---

## References

1. Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model. *Nature*. 2025. doi:10.1038/s41586-024-08328-6
2. Grinsztajn L, Flöge K, Key O, et al. TabPFN-2.5: Advancing the state of the art in tabular foundation models. arXiv:2511.08667. 2025.
3. Grinsztajn L, Flöge K, Key O, et al. TabPFN-3: Technical report. arXiv:2605.13986. 2026.
4. Prior Labs. TabPFN-3.5: Technical report. 2026. https://priorlabs.ai/technical-reports/tabpfn-3-5 ; software `tabpfn` 9.0.0, https://github.com/PriorLabs/TabPFN
5. Qu J, Holzmüller D, Varoquaux G, Le Morvan M. TabICLv2: A better, faster, scalable, and open tabular foundation model. *ICML*. 2026. arXiv:2602.11139.
6. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks Track*. 2022.
7. McElfresh D, Khandagale S, Valverde J, et al. When do neural nets outperform boosted trees on tabular data? *NeurIPS Datasets and Benchmarks Track*. 2023.
8. Herre M, Tschalzev A, Marton S, Bartelt C. Explaining tabular foundation model differences through meta-features. arXiv:2605.28418. 2026.
9. Wu Y, Wang A. Information rather than architecture: cross-learner attribution and clinician–model complementarity in CIN2+ risk prediction for women referred for colposcopy. *Frontiers in Oncology*. Manuscript under review. 2026.
10. Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*. 1982;143:29–36.
11. Hill MO. Diversity and evenness: a unifying notation and its consequences. *Ecology*. 1973;54:427–432.
12. Nori H, Jenkins S, Koch P, Caruana R. InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223. 2019.
13. Prokhorenkova L, Gusev G, Vorobev A, Dorogush AV, Gulin A. CatBoost: unbiased boosting with categorical features. *NeurIPS*. 2018.
14. Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS*. 2017.
15. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. *KDD*. 2016.
16. Breiman L. Random forests. *Machine Learning*. 2001;45:5–32.
17. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting deep learning models for tabular data. *NeurIPS*. 2021.
