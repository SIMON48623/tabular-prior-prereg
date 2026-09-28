# Phase 1, Section 0: frozen driver and pre-run checks

This folder is committed before the first model fit on any pool dataset, as Section 5.1 of the
registered plan requires. It does not change any registered file: the registered files and their
hashes remain exactly those in `../MANIFEST.sha256` (commit `68d5d00`).

## Contents

| Path | Content |
|---|---|
| `phase1_driver/`, `phase1_driver_sha256.txt` | The frozen Phase 1 driver (35 files) and the SHA-256 of each file. It imports the registered model code from `pipeline_code/` unchanged, except for D1 |
| `DEVIATIONS.md` | Deviations from the registered plan (D1, D2) and implementation notes |
| `smoke_phase1.log` | Production-mode smoke test of the frozen driver on the RTX 4090: 10 models × 2 toy variants, all `ok`, offline, no performance quantity. Every record carries the driver manifest hash |
| `parameter_hash_check.md` | Hashes of the loaded foundation-model parameters, computed in two fresh processes on different inputs without fitting; they are identical |
| `lr_reference_214.csv` | Admission of the second CPU instance: its reference logistic-regression AUROCs equal the pre-flight values on all 478 datasets |
| `host_assignment.csv` | Assignment of each dataset to one CPU instance, balanced by projected work |
| `marginal_check.csv` | The driver's Phase 0b marginal-AUROC path reproduces `final_pool/marginal_aucs.json` on all 478 datasets (largest difference 0) |
| `injection_sanity.csv` | Model-free check of the injected signal columns (brief, M2 step 4) |
| `records/` | Change notes of each review round, the fixed-output dry run on pool data, and the fault-injection test |

## Key hashes

| Item | SHA-256 |
|---|---|
| `phase1_driver_sha256.txt` | `58d0f8527a563f8dc849cba8fb448b8661f8c39d2a6c92468d6639cf09835839` |
| `smoke_phase1.log` | `09d3335a156690b2d6683da45c3d2a8aeb02d79a9ae4bd16a183c10b5073b18f` |
| cleaned-data per-file ledger (identical on all three machines) | `a807db2390d5a85a9b77f1286e6ab7d1aa1a1f7faaf2396f3be8459220e0ce68` |
| Loaded parameters, TabPFN-3.5 | `6ac4478f8eb2c5674b5273b96399617b915321bfaefae660d99615cf34c785aa` |
| Loaded parameters, TabICLv2 | `a0842ed20731a708a88ce9c1ebcc6666e3f02d17008ebd8148e6a4f55d1e44d8` |
| Loaded parameters, TabPFN-2 | `f1fa31642be5efb42621b576450d8390d21f94b49012454977924bc2527a74f2` |

`SECTION0_MANIFEST.sha256` lists the SHA-256 of every file in this folder. To check, from this folder:

```bash
sha256sum -c SECTION0_MANIFEST.sha256
sha256sum -c phase1_driver_sha256.txt
```

Every Phase 1 checkpoint records the driver manifest hash above, so each result can be traced to this
exact code.
