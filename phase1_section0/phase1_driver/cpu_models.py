from __future__ import annotations

import importlib.util
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def xgb_d1_fit_predict(
    pilot: Any,
    train: pd.DataFrame,
    labels: np.ndarray,
    test: pd.DataFrame,
    categorical: list[str],
    input_audit: Callable[[Any], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Post-registration deviation D1.

    This is the registered XGBoost branch with one deliberate difference:
    categorical metadata on held-out rows is fixed to the training categories,
    so unseen ordinal values (including -1) become missing values.
    """
    from xgboost import XGBClassifier

    _lr_prep, ord_prep, n_cat = pilot.preprocessors(list(train.columns), categorical)
    if not n_cat:
        output, _fit_seconds, _predict_seconds = pilot.cpu_fit_predict(
            "xgb", train, labels, test, categorical
        )
        return np.asarray(output, dtype=float), 0
    Xtr = np.asarray(ord_prep.fit_transform(train), dtype=float)
    if input_audit is not None:
        input_audit(Xtr)
    cat_idx = list(range(n_cat))
    model = XGBClassifier(random_state=pilot.SEED, n_jobs=1, enable_categorical=True)
    Xtr_xgb = pd.DataFrame(Xtr)
    for idx in cat_idx:
        Xtr_xgb[idx] = Xtr_xgb[idx].astype(int).astype("category")
    with threadpool_limits(limits=1):
        model.fit(Xtr_xgb, labels)
    Xte = np.asarray(ord_prep.transform(test), dtype=float)
    if input_audit is not None:
        input_audit(Xte)
    Xte_xgb = pd.DataFrame(Xte)
    unseen_cells = 0
    for idx in cat_idx:
        values = Xte_xgb[idx].astype(int)
        categories = Xtr_xgb[idx].cat.categories
        known = values.isin(categories)
        unseen_cells += int((~known).sum())
        Xte_xgb[idx] = pd.Categorical(values.where(known, np.nan), categories=categories)
    with threadpool_limits(limits=1):
        pred = model.predict_proba(Xte_xgb)[:, 1]
    return np.asarray(pred, dtype=float), unseen_cells


def fit_predict(model_name: str, train: pd.DataFrame, labels: np.ndarray,
                test: pd.DataFrame, categorical: list[str], registered_root: Path,
                input_audit: Callable[[Any], None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    random.seed(13)
    np.random.seed(13)
    if model_name == "lr":
        phase0b = _load(
            registered_root / "pipeline_code" / "phase0b" / "phase0b_experiment.py",
            "phase0b_frozen_driver",
        )
        declared = {column: column in categorical for column in train.columns}
        combined = pd.concat([train, test], axis=0, ignore_index=True)
        prepared, numeric, categories = phase0b.prepare_feature_frame(combined, declared)
        train_prepared = prepared.iloc[:len(train)]
        test_prepared = prepared.iloc[len(train):]
        pipeline = phase0b.build_lr_pipeline(numeric, categories)
        if input_audit is not None:
            audit_pipeline = phase0b.build_lr_pipeline(numeric, categories)
            transformed_train = audit_pipeline.named_steps["preprocess"].fit_transform(train_prepared)
            transformed_test = audit_pipeline.named_steps["preprocess"].transform(test_prepared)
            input_audit(transformed_train)
            input_audit(transformed_test)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            pipeline.fit(train_prepared, labels)
            output = pipeline.predict_proba(test_prepared)[:, 1]
        parameter = "frozen phase0b build_lr_pipeline; threadpool_limits=1"
        positions = [train.columns.get_loc(c) for c in categorical]
    else:
        pilot = _load(
            registered_root / "pipeline_code" / "phase0c" / "phase1_pilot.py",
            "phase1_pilot_frozen_driver",
        )
        if model_name == "xgb" and categorical:
            output, unseen_cells = xgb_d1_fit_predict(
                pilot, train, labels, test, categorical, input_audit=input_audit
            )
            return output, {
                "categorical_positions": list(range(len(categorical))),
                "categorical_parameter": (
                    "D1: test categories fixed to training categories; unseen ordinal values -> NaN"
                ),
                "unseen_category_handling": "D1 training categories; unseen -> NaN",
                "unseen_category_cells": unseen_cells,
            }
        if input_audit is not None:
            _lr_prep, ordinal, _n_cat = pilot.preprocessors(list(train.columns), categorical)
            transformed_train = np.asarray(ordinal.fit_transform(train), dtype=float)
            transformed_test = np.asarray(ordinal.transform(test), dtype=float)
            input_audit(transformed_train)
            input_audit(transformed_test)
        output, _fit_seconds, _predict_seconds = pilot.cpu_fit_predict(
            model_name, train, labels, test, categorical,
        )
        positions = list(range(len(categorical)))
        parameter = {
            "ebm": f"feature_types nominal at {positions}",
            "catboost": f"cat_features={positions}; encoded categories converted to strings",
            "lgbm": f"categorical_feature={positions}",
            "xgb": f"enable_categorical=True; category dtype at {positions}",
            "rf": "ordinal encoded; no categorical parameter",
        }[model_name]
    return np.asarray(output, dtype=float), {
        "categorical_positions": positions,
        "categorical_parameter": parameter,
        "unseen_category_handling": {
            "lr": "OneHotEncoder(handle_unknown=ignore)",
            "ebm": "OrdinalEncoder unknown_value=-1; nominal feature",
            "catboost": "OrdinalEncoder unknown_value=-1; converted to string category",
            "lgbm": "OrdinalEncoder unknown_value=-1; categorical_feature",
            "xgb": "registered path (no categorical columns)",
            "rf": "OrdinalEncoder unknown_value=-1",
        }[model_name],
    }
