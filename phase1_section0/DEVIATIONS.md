# Deviations from the registered plan

Registration: OSF, DOI 10.17605/OSF.IO/GAFDR; repository commit
`68d5d00916ca2af60fdda4e7272bd416611aa531` (2026-09-27 18:49:05 UTC).

Everything below was decided and committed before the first model fit on any pool dataset. At that
time no model-comparison quantity existed. Each item will be reported in the manuscript, as Section 11
of the plan requires.

## D1. XGBoost: categorical levels in held-out data that are absent from training

**Registered.** Phase 1 imports the conventional models' preprocessing and fitting unchanged from
`pipeline_code/` (plan Section 5.1). For XGBoost this is the `xgb` branch of
`pipeline_code/phase0c/phase1_pilot.py::cpu_fit_predict`. Categorical columns are integer-encoded by
`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, and XGBoost receives them as a
pandas `category` dtype with `enable_categorical=True` (Section 5.2).

**Problem.** The registered function builds the held-out fold's `category` dtype from the held-out
values alone. With the registered xgboost 3.4.1, a held-out level that does not occur in the training
data (encoded as −1) makes prediction fail: `XGBoostError: Found a category not in the training set`.
The registered pilot used numeric data only, so this path was never exercised before registration.
The driver smoke test found it on a toy dataset built to contain such a level. Left unchanged, XGBoost
would fail on every outer and inner fold that contains a rare level, drop out of the inner selection
there, and weaken the comparator on datasets with categorical columns.

**Change.** The driver adds one function, `phase1_driver/cpu_models.py`. It is identical to the
registered `xgb` branch except for one step: the held-out (and inner validation) `category` dtype
takes the categories of the training data, and values not seen in training, including −1, become
missing. XGBoost then routes them by its learned default direction for missing values. Datasets
without categorical columns call the registered function unchanged. The registered file is not
modified.

**Verification.** Whenever no held-out level is unseen in training, the new function gives
predictions bitwise identical to the registered function. The tests cover a held-out fold that lacks
the highest level and one that lacks a middle level. The function therefore differs from the
registered path only where the registered path fails.

## D2. Execution safeguards: time limit per work unit

**Registered.** A model that fails after one retry (on GPU, for the foundation models) is recorded as
failed with its reason (plan Section 6.8; brief Section 7). The plan sets no time limit.

**Change.** Each work unit (one dataset, variant, outer fold and machine group) has a time limit of
ten times its projected run time, and at least one hour. The projected time comes from the registered
timing model (`phase0c_outputs/compute_projection.md`) scaled by the pre-flight EBM time ratio. A unit
that exceeds the limit, or whose worker process crashes, is attempted once more in a new process. If
it fails a second time, every model in that unit is recorded as failed with the reason, following
Section 6.8. Every such event is kept in the run log and reported with the results.

**Why.** Without a limit, one hung fit would stall the run. The limit is generous by design: a fit
that is merely slow still completes.

## Implementation notes (not deviations)

- **M6 input.** `cleaned_data` stores categorical columns as integer codes. Before calling the
  registered Phase 0b marginal-AUROC functions, the driver decodes them to the original string
  levels, so the functions receive the same input as in Phase 0b. This reproduces the frozen marginal
  AUROCs of all 478 datasets exactly (`marginal_check.csv`, largest difference 0).
- **FT-Transformer.** It follows the registered input specification (one-hot encoding plus
  standardisation), so a dataset with many categorical levels gives one token per one-hot column.
  Datasets where this exceeds GPU memory are recorded as FT-Transformer failures under Section 6.8.
  FT-Transformer enters only exploratory analysis E5.
- **GPU instance.** The foundation models run on a cloned RTX 4090 instance (`host_registry.json`).
  This is the same GPU type the plan registers.
- **Injection sanity check** (brief, M2 step 4). 35 of 2,856 injected-column checks differ from the
  target level by more than 0.06 (`injection_sanity.csv`). They come from 17 datasets, 16 of which have
  at most 705 rows (the target-region median is 1,471), where sampling variation is expected. The brief
  says to record them and not to stop.
