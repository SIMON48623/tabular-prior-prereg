# Phase 1 run records

Records made during the Phase 1 run, after the frozen driver started (commit `b7ba954`). They do
not change any registered file, and the frozen driver is unchanged.

| Path | Content |
|---|---|
| `DEVIATIONS_phase1_run.md` | D3: albert and Amazon employee access families excluded (EBM run time). D4: units that failed by time-out are rerun without a time limit |
| `incident_2026-09-29_restart.md` | Execution record of the CPU scheduler restart on 2026-09-29 |
| `evidence/phase1_diag_timeout.zip` | Timing-only diagnostics: single-fit times per model (44708, 880), EBM iteration counts, categorical-level screen of all 478 datasets, time-out events up to 2026-09-29 01:28 UTC |
| `evidence/m1_timing_check.zip` | Elapsed time and time limit of every completed M1 conventional-model unit (2,362 units) |
| `evidence/d3_restart_diag_*_redacted.zip` | Diagnostics of the failed restart. `proc_comparison_*.json` was removed from each, because it contains the cloud provider's service URLs and internal addresses |
| `evidence/health_15min_*.json`, `evidence/restart_round_summary.md` | Checks of the final restart |
| `SHA256SUMS.txt` | SHA-256 of every file in this folder |

None of these files contains a performance quantity.

The archives as received have these SHA-256 values:
- `phase1_diag_timeout.zip`: `6a3b24e7ef7a944063f13c41246dbc2f6b2317ab8aa0e657ed3def08689dca09`
- `m1_timing_check.zip`: `80051dbc550ea36a4e6d3f7bbba12ff61785ed0037d69abf0fd16639fcadb0a7`
- `d3_restart_diag_214.zip`, before redaction: `fc17e83e24c62e68eeba0317893d8d1c32b24fccd5dabf2bd4ea3743a4bcaf83`
- `d3_restart_diag_686.zip`, before redaction: `eac850814a4bd6642c2da7e2103e29f5edc8d483123e2f6208af0c4d4e809a37`
