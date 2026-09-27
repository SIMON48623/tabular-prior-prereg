# Phase 1 compute projection

This projection uses the complete Phase 0c pilot timing table. For each model and stage (outer/inner), a log-linear least-squares model of total fit-plus-predict seconds against log training rows and log feature count was fitted, then evaluated for every final-pool dataset and registered module workload.

- Final pool: 478 datasets; target region (`0.60 <= auc_lr <= 0.85`): 238 datasets.
- CPU: `Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz`; current-server traditional-model capacity: 16 concurrent single-thread workers.
- GPU: `NVIDIA GeForce RTX 4090`; one sequential GPU worker.
- M2 assumes the default three injection repetitions (39 variants per target dataset).
- M3 assumes all 12 registered removal variants; every target dataset has at least five retained features.
- M4 eligibility is evaluated per dataset from its frozen folds, sample size, and minority rate.
- M5 includes 192 registered grid cells × 20 repetitions = 3,840 cases.
- M6 fits no model; the tables therefore report zero traditional-model CPU core-hours and zero GPU hours for M6. Its marginal-AUROC bookkeeping cost is outside the model-fit projection.

## Projected resource time

| Module | Single-thread CPU core-hours | GPU hours |
|---|---:|---:|
| M1 | 419.24 | 15.17 |
| M2 | 10943.55 | 48.63 |
| M3 | 2177.13 | 14.03 |
| M4 | 2357.04 | 16.56 |
| M5 | 829.58 | 3.92 |
| M6 | 0.00 | 0.00 |
| **Total** | **16726.54** | **98.31** |

## Projected resource time by model

| Model | Resource | Hours | Median pilot fit seconds |
|---|---|---:|---:|
| catboost | CPU | 775.56 | 5.429831 |
| ebm | CPU | 15855.33 | 112.884236 |
| lgbm | CPU | 27.50 | 0.123979 |
| lr | CPU | 1.66 | 0.008328 |
| rf | CPU | 41.91 | 0.241888 |
| xgb | CPU | 24.59 | 0.139845 |
| ftt | GPU | 1.30 | 2.120944 |
| tabicl2 | GPU | 17.87 | 0.519169 |
| tabpfn2 | GPU | 12.28 | 0.183198 |
| tabpfn35 | GPU | 66.86 | 1.847606 |

## Wall-clock scenarios

Within each module, CPU and GPU work are assumed to run concurrently; modules are summed in registered sequence. Additional 32-core instances are assumed to have equivalent per-core throughput and perfect task distribution.

| Additional 32-core CPU instances | Available traditional-model workers | Wall hours | Wall days |
|---:|---:|---:|---:|
| 0 | 16 | 1045.41 | 43.56 |
| 1 | 48 | 354.91 | 14.79 |
| 2 | 80 | 219.01 | 9.13 |
| 4 | 144 | 128.61 | 5.36 |
| 8 | 272 | 98.31 | 4.10 |

## EBM share

Projected EBM time is 15855.33 single-thread core-hours, 94.79% of all projected traditional-model CPU core-hours.

These are workload projections from the fixed pilot and final-pool size distribution, not performance comparisons.
