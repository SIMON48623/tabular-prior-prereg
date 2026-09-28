from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


@dataclass(frozen=True)
class OrdinalPrepared:
    train: np.ndarray
    test: np.ndarray
    categorical_positions: list[int]
    transformer: ColumnTransformer


@dataclass(frozen=True)
class FTTPrepared:
    train: np.ndarray
    test: np.ndarray
    cat_cardinalities: list[int]
    transformer: ColumnTransformer


@dataclass(frozen=True)
class TabICLPrepared:
    train: pd.DataFrame
    test: pd.DataFrame
    categorical_positions: list[int]
    transformer: ColumnTransformer


def _columns(frame: pd.DataFrame, categorical: list[str]) -> tuple[list[str], list[str]]:
    unknown = set(categorical) - set(frame.columns)
    if unknown:
        raise ValueError(f"unknown categorical columns: {sorted(unknown)}")
    return [c for c in frame.columns if c not in categorical], list(categorical)


def prepare_ordinal_views(train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]) -> OrdinalPrepared:
    numeric, categorical = _columns(train, categorical)
    parts = []
    if categorical:
        parts.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical))
    if numeric:
        parts.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), numeric))
    transformer = ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
    train_array = np.asarray(transformer.fit_transform(train), dtype=float)
    test_array = np.asarray(transformer.transform(test), dtype=float)
    if np.isnan(train_array).any() or np.isnan(test_array).any():
        raise RuntimeError("NaN reached model boundary after ordinal preparation")
    return OrdinalPrepared(train_array, test_array, list(range(len(categorical))), transformer)


def prepare_tabicl_views(train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]) -> TabICLPrepared:
    prepared = prepare_ordinal_views(train, test, categorical)
    columns = [f"feature_{position}" for position in range(prepared.train.shape[1])]
    train_out = pd.DataFrame(prepared.train, columns=columns)
    test_out = pd.DataFrame(prepared.test, columns=columns)
    for position in prepared.categorical_positions:
        column = columns[position]
        categories = sorted(set(train_out[column].tolist()) | {-1.0})
        train_out[column] = pd.Categorical(train_out[column], categories=categories)
        test_out[column] = pd.Categorical(test_out[column], categories=categories)
    if train_out.isna().any().any() or test_out.isna().any().any():
        raise RuntimeError("NaN reached model boundary after TabICL ordinal preparation")
    return TabICLPrepared(
        train_out, test_out, prepared.categorical_positions, prepared.transformer
    )


def prepare_ftt_views(train: pd.DataFrame, test: pd.DataFrame, categorical: list[str]) -> FTTPrepared:
    numeric, categorical = _columns(train, categorical)
    parts = []
    if numeric:
        parts.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        parts.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    transformer = ColumnTransformer(parts, remainder="drop")
    train_value = transformer.fit_transform(train)
    test_value = transformer.transform(test)
    train_array = np.asarray(train_value.toarray() if hasattr(train_value, "toarray") else train_value, dtype=np.float32)
    test_array = np.asarray(test_value.toarray() if hasattr(test_value, "toarray") else test_value, dtype=np.float32)
    if np.isnan(train_array).any() or np.isnan(test_array).any():
        raise RuntimeError("NaN reached model boundary after FT-Transformer preparation")
    return FTTPrepared(train_array, test_array, [], transformer)
