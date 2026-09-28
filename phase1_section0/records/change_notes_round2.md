# Phase 1 driver review changes

The accepted 214 admission result and `host_assignment.csv` were preserved and
were not recomputed.

## Required corrections

- Added automatic registered-input, driver-manifest, generator, lock,
  assignment, host/hardware and local-weight verification. Foundation models
  receive explicit registered local paths, offline hub variables and a socket
  guard. A failed attestation or weight load cannot write a checkpoint.
- Made TabICL consume the exact TabPFN ordinal view, represented as a DataFrame
  with categorical columns typed as pandas `category`.
- Changed prediction checkpoint storage from row dictionaries to column arrays.
  Aggregation now treats `build_plan()` as the authority, requires exactly one
  matching checkpoint per unit, rejects extras, missing units, identity
  mismatches and both registered duplicate-key forms, and streams by module and
  dataset.
- Added checkpoint run records and per-host environment records, including
  elapsed time, status/failures, undefined S, inner failures, actual weight
  hashes, offline variables and deterministic settings.
- Added the 32/32/1 host schedulers. They establish `PYTHONHASHSEED=13`, single
  traditional-model thread variables and host labels before starting Python.
- Probability validation is now inside retry handling. CUDA context/OOM/driver
  failures escape as process-fatal errors without checkpoint creation.
- FT-Transformer validation/test forward passes are batched and its sigmoid is
  evaluated in float64.
- M6 now reverses registered parquet category codes to original Phase 0b string
  levels, reconstructs the Phase 0b label coding, and then calls the registered
  `prepare_feature_frame` and `single_feature_percentile_scores` functions.
- Added `marginal_check.csv` and `injection_sanity.csv` model-free checks.
- Defined `row_index` strictly as parquet row position and removed the silent
  fallback.
- Extended smoke coverage to actual input matrices, missing/unseen category
  levels and XGBoost equivalence; no performance metric is produced.
- Fixed M5 retry state, CPU/foundation inner retry, M1 foundation-selection
  directory creation, the 238-row frozen target-region source, injection repeat
  assertion and failed-model prediction-count validation.

## D1 — XGBoost post-registration deviation

The frozen registered XGBoost path errors when a held-out category is absent
from training. The registered file remains unchanged. The driver adds one
scoped XGBoost function whose sole algorithmic difference is that held-out
categorical dtypes use training categories and unseen values become `NaN`.
Datasets without categorical columns still call the registered function.

On the toy no-unseen fixture, D1 is bitwise identical to the registered path.
On the unseen fixture, D1 completes and logs the mapping of unseen ordinal
values (including `-1`) to missing. This result is enforced by a unit test and
is repeated in the final 4090 smoke run.

## Model-free verification result

- `marginal_check.csv`: 10,808 feature rows across all 478 datasets; maximum
  absolute difference from `final_pool/marginal_aucs.json` is `0.0`.
- `injection_sanity.csv`: 2,856 rows (238 datasets × 3 repetitions × 4 levels).
  Deviations of at least 0.06 are recorded as required and are not a stop
  condition.
