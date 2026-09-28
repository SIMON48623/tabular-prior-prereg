from __future__ import annotations

import os
import random
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

SEED = 13
INJECT_REPS = 3
CPU_MODELS = ("lr", "ebm", "catboost", "lgbm", "xgb", "rf")
FOUNDATION_MODELS = ("tabpfn35", "tabicl2", "tabpfn2")
ALL_MODELS = FOUNDATION_MODELS + CPU_MODELS + ("ftt",)
EXECUTION_STAGES = (
    "M1", "M6", "M2_PRIORITY_R0", "M2_PRIORITY_R1", "M2_PRIORITY_R2",
    "M4", "M2_REMAINDER", "M5",
)
PREDICTION_COLUMNS = ("openml_id", "variant", "fold", "row_index", "y", "model", "p")
SELECTION_COLUMNS = (
    "id", "variant", "fold", "model", "inner_auroc_1", "inner_auroc_2",
    "inner_auroc_3", "inner_auroc_4", "inner_auroc_5", "inner_auroc_mean",
    "inner_status", "selected",
)
METRIC_COLUMNS = (
    "id", "variant", "fold", "model", "auroc", "logloss", "brier", "n_test",
    "n_pos_test", "n_train", "fit_seconds", "host", "cpu_model", "gpu_model",
    "n_threads", "status", "failure_reason",
)


def m2_priority_variants() -> list[str]:
    levels = ("0.65", "0.75", "0.85", "0.95")
    return [
        f"{arm}_{level}_r{rep}"
        for rep in range(INJECT_REPS)
        for arm in ("conc", "dispn")
        for level in levels
    ]


def m2_remainder_variants() -> list[str]:
    levels = ("0.65", "0.75", "0.85", "0.95")
    return [variant for rep in range(INJECT_REPS) for variant in (
        f"noise_r{rep}", *(f"disp_{level}_r{rep}" for level in levels)
    )]


def gpu_determinism_record() -> dict[str, Any]:
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    common = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "seed": SEED,
    }
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {
            **common,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "deterministic_algorithms": True,
            "torch_applied": False,
            "torch_import_error": f"{type(exc).__name__}: {exc}",
        }
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return {
        **common,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_applied": True,
    }
