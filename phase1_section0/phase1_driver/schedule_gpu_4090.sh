#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=13
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
[ "$PYTHONHASHSEED" = 13 ] && [ "$OMP_NUM_THREADS" = 1 ] && [ "$MKL_NUM_THREADS" = 1 ] && [ "$OPENBLAS_NUM_THREADS" = 1 ]
WORKERS=1
exec python -m phase1_driver.scheduler --group gpu --workers "$WORKERS" "$@"
