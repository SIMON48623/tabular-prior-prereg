# Dispersion of predictive signal and the value of tabular foundation models: preregistration v2

This repository is the frozen record of the preregistration. The plan is in
[`docs/preregistration_v2.md`](docs/preregistration_v2.md). Author: Yuzhang Wu.

**Status at registration (2026-09-27).** The only model fitted to any dataset in the pool is the
reference logistic regression of Phase 0b. No model-comparison quantity has been computed on pool
data. Phase 1, which fits the foundation models and the conventional comparators, has not started.
Section 10 of the preregistration lists everything that has been observed.

All datasets are public OpenML classification datasets: 478 in the final pool, in 188 families.

## Layout

| Path | Content |
|---|---|
| `docs/preregistration_v2.md` | The registered plan, filled from the frozen files by `code/fill_prereg.py` |
| `docs/preregistration_v2.template.md` | The plan with placeholders; `fill_prereg.py` reads it |
| `docs/phase*_brief.md` | Instructions given to the executor, phase by phase, including the pre-flight and Phase 1 |
| `code/` | Frozen code (generators, stratifier, estimators, pool merge), its tests, and the build, calibration, construct-check, threshold, power and fingerprint scripts |
| `code/exploration/` | Informal explorations that preceded the frozen stratifier (record only; see its README) |
| `phase0b_outputs/` | Phase 0b pool of 364 datasets: stratifiers, marginal AUROCs, environment, report and log |
| `phase0c_outputs/` | Phase 0c outputs (21 files): extended pool, reproduction checks, alias candidates, metadata, alternative stratifiers, Phase 1 environment and lock file, smoke test, synthetic pilot, compute projection, logistic-regression reference |
| `phase0c_history/` | Outputs of the two Phase 0c runs that stopped on their stop rules (Section 3.3) |
| `pool_review/` | The same-size alias screen behind the family review |
| `final_pool/` | The merged final pool (`merge_pool.py` output) |
| `frozen/` | Pool decisions, medical-subset labels, CPU instances and scope decision (`compute_plan.json`), stratifier values and cutpoints, removal plan, manipulation checks, construct check, threshold comparison, generator fingerprints |
| `power/` | Power simulations |
| `preflight_outputs/` | Pre-flight on the Phase 1 CPU type: logistic-regression reference, timing, CPU record, report |
| `pipeline_code/`, `pipeline_code_sha256.txt` | The executor's Phase 0b and Phase 0c code, with the SHA-256 of each file; Phase 1 imports its model, preprocessing and fitting functions unchanged |
| `MANIFEST.sha256` | SHA-256 of every file above (written by `code/make_manifest.py`) |

`cleaned_data.zip`, the cleaned data read by Phase 1, is not committed because of its size. Its
SHA-256 is `93d97f647acc7ebbdde1e55afc95c5cd55eb5cb9fd814f3070356676b724c646`; Phase 1 checks it
before use. `phase0c_outputs/` is the unmodified content of the executor's archive
`phase0c_outputs.zip` (SHA-256 `e5e115091619155518f734579a8e7ae59b5edb79b76a271350db5d870225c8cc`).

## Checking the record

Requires Python 3.12 with numpy, pandas, scipy and scikit-learn. From the repository root:

```bash
sha256sum -c MANIFEST.sha256
(cd pipeline_code && sha256sum -c ../pipeline_code_sha256.txt)

python3 code/test_generators.py
python3 code/test_stats_core.py
python3 code/test_merge_pool.py
python3 code/test_phase0c_functions.py

# Rebuild the derived files; each must be byte-identical to the committed one
python3 code/merge_pool.py phase0b_outputs phase0c_outputs frozen/pool_decisions.json /tmp/final_pool_check
diff -r /tmp/final_pool_check final_pool
python3 code/build_frozen_stratifier.py
python3 code/build_frozen_plans.py
python3 code/fill_prereg.py
sha256sum -c MANIFEST.sha256
```

Before the first model fit on any pool dataset, a later commit adds the Phase 1 driver
(`phase1_driver/`), its SHA-256 and its smoke-test log (Section 5.1 of the plan).

`python3 code/make_manifest.py` rewrites `MANIFEST.sha256` from the registered files; it refuses to run
while the plan is still marked as waiting for the pre-flight. Only the files listed in the manifest are
committed (`.gitignore` keeps working files out).

The power simulations (`code/power_sim.py`) and the construct and threshold checks take longer and
write `power/power_results.json` and the corresponding files in `frozen/`.
