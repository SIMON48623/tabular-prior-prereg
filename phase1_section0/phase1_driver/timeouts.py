from __future__ import annotations

import importlib.util
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from .execution import WorkUnit


MINIMUM_UNIT_TIMEOUT_SECONDS = 3600.0
TIMEOUT_MULTIPLIER = 10.0
REGISTERED_PREFLIGHT_EBM_RATIO = 2.765933


def timeout_from_projection_seconds(projected_seconds: float, ebm_ratio: float) -> float:
    if projected_seconds < 0 or not math.isfinite(projected_seconds):
        raise ValueError("projected_seconds must be finite and nonnegative")
    if ebm_ratio <= 0 or not math.isfinite(ebm_ratio):
        raise ValueError("ebm_ratio must be finite and positive")
    return max(
        MINIMUM_UNIT_TIMEOUT_SECONDS,
        float(projected_seconds) * float(ebm_ratio) * TIMEOUT_MULTIPLIER,
    )


def _load_registered_projection(root: Path) -> Any:
    candidates = (
        root / "pipeline_code" / "phase0c" / "compute_projection.py",
        root / "compute_projection.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("registered compute_projection.py not found")
    spec = importlib.util.spec_from_file_location("phase1_registered_compute_projection", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import registered compute projection: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def _projection_models(root_text: str) -> tuple[Any, dict[tuple[str, str], np.ndarray]]:
    root = Path(root_text)
    module = _load_registered_projection(root)
    timing = pd.read_csv(root / "phase0c_outputs" / "pilot_timing.csv")
    fit = getattr(module, "fit_log_models", None) or getattr(module, "_fit_log_models", None)
    if fit is None:
        raise RuntimeError("registered compute projection lacks fit_log_models")
    return module, fit(timing)


def _predict(module: Any, models: dict[tuple[str, str], np.ndarray],
             model: str, stage: str, n: float, p: float) -> float:
    predict = getattr(module, "predict", None) or getattr(module, "_predict_cost", None)
    if predict is None:
        raise RuntimeError("registered compute projection lacks predict")
    return float(predict(models, model, stage, n, p))


@lru_cache(maxsize=4096)
def _pool_shape(root_text: str, identifier: str, fold: int) -> tuple[int, int]:
    frame = pd.read_parquet(Path(root_text) / "cleaned_data" / f"{identifier}.parquet")
    features = [name for name in frame.columns if name not in {"y", "fold"}]
    return int((frame["fold"].to_numpy() != int(fold)).sum()), len(features)


def projected_unit_seconds(root: Path, unit: WorkUnit) -> float:
    if unit.module == "M6" or unit.group == "none":
        return 0.0
    module, models = _projection_models(str(root.resolve()))
    if unit.module == "M5":
        n = int(unit.variant.split("_", 1)[0][1:])
        p = 32
    else:
        n, p = _pool_shape(str(root.resolve()), unit.identifier, unit.fold)
        if unit.module == "M2":
            p += 15 if unit.variant.startswith("dispn_") else 8
        elif unit.module == "M4":
            n = int(unit.variant.split("_", 1)[0][1:])
    inner_n = 0.8 * n
    if unit.group == "cpu":
        return sum(
            _predict(module, models, model, "outer", n, p)
            + 5.0 * _predict(module, models, model, "inner", inner_n, p)
            for model in config.CPU_MODELS
        )
    if unit.group == "gpu":
        inner_multiplier = 5.0 if unit.module == "M1" else 0.0
        return sum(
            _predict(module, models, model, "outer", n, p)
            + inner_multiplier * _predict(module, models, model, "inner", inner_n, p)
            for model in config.FOUNDATION_MODELS
        )
    if unit.group == "ftt":
        return _predict(module, models, "ftt", "outer", n, p)
    raise RuntimeError(f"unsupported unit group for timeout: {unit.group}")


def unit_timeout_seconds(root: Path, unit: WorkUnit) -> float:
    return timeout_from_projection_seconds(
        projected_unit_seconds(root, unit), REGISTERED_PREFLIGHT_EBM_RATIO,
    )

