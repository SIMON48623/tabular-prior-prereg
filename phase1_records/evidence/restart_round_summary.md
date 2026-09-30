# D3 incident-fix restart round

## Original-environment checks

- Host 214: original scheduler PID 19898; recorded working directory `/root/autodl-tmp/phase1_c1_verify_136c1e42`; recorded environment contained 33 variables; `TABPFN_TOKEN` was absent; the executed Python path and the Python path resolved from the recorded `PATH` were both `/root/autodl-tmp/tabular_prior_phase1_setup/env_phase1_exact/bin/python`.
- Host 686: original scheduler PID 36390; recorded working directory `/root/autodl-tmp/phase1_c1_verify_136c1e42`; recorded environment contained 33 variables; `TABPFN_TOKEN` was absent; the executed Python path and the Python path resolved from the recorded `PATH` were both `/root/autodl-tmp/tabular_prior_phase1_preflight/env_phase1_exact/bin/python`.

## Pre-start checks

- Host 214: related process count 0; process-open checkpoint lock count 0; `/proc/locks` checkpoint lock count 0.
- Host 686: related process count 0; process-open checkpoint lock count 0; `/proc/locks` checkpoint lock count 0.
- Host 686 verify-only: return code 0; `status=verified`; `driver_manifest_sha256=136c1e4207c5168be34288ac4846aae147bc4234ba4e566f23d8adcdfd0cc68c`.
- The immediately preceding host-214 verify-only had return code 0, `status=verified`, and the same driver-manifest SHA-256.

## Restart

- Host 214: scheduler PID 295108; start Unix time 1790687584.2871516; command line matched the recorded original command line; working directory matched the recorded original working directory; `scheduler_failures.jsonl` baseline was 76 lines.
- Host 686: scheduler PID 336826; start Unix time 1790687585.108943; command line matched the recorded original command line; working directory matched the recorded original working directory; `scheduler_failures.jsonl` baseline was 62 lines.

## Fifteen-minute health check

- Host 214 at 931.7885608673096 seconds: scheduler alive; workers 32; worker-held checkpoint locks 32; failure-line delta 0; traceback absent.
- Host 686 at 928.0932381153107 seconds: scheduler alive; workers 32; worker-held checkpoint locks 32; failure-line delta 0; traceback absent.
- Host 214's interactive SSH connection closed after its final sample was returned. No signal was sent to the scheduler or its workers.
- Host 4090 was not modified.
