from __future__ import annotations

import ast
import json
import math
import os
import platform
import re
import shutil
import time
import traceback
import warnings
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUTPUT_PARENT = ROOT / "outputs"
OUTPUT = OUTPUT_PARENT / "phase0b_outputs"
CANDIDATES = WORK / "phase0b_candidate_pool.csv"
CHECKPOINT = WORK / "phase0b_checkpoint.csv"
MARGINAL_CHECKPOINT_DIR = WORK / "phase0b_marginals"
SELF_CHECK_PATH = WORK / "phase0b_self_check.json"
OPENML_CACHE = WORK / "openml_cache"

SEED = 42
N_SPLITS = 5
MAX_WORKERS = 3
DOWNLOAD_ATTEMPTS = 3
N_MIN = 200
N_MAX = 10_000
P_MIN = 5
P_MAX = 100
MINORITY_RATE_MIN = 0.05
MINORITY_RATE_MAX = 0.50
MAX_COLUMN_MISSING_RATE = 0.30
NO_SIGNAL_AUC_LR_MAX = 0.52
TARGET_REGION_MIN = 0.60
TARGET_REGION_MAX = 0.85

# Same operational structure screen as Phase 0, expanded before any Phase 0B
# outcome calculation to cover explicit time-series resampling names in the pool.
TIME_SERIES_NAME_RE = re.compile(
    r"(?:rmftsa|pm10|pbcseq|delta[_-](?:ailerons|elevators)|chatfield|"
    r"el[_-]?nino|(?:^|[_-])stock(?:$|[_-])|(?:^|[_-])wind(?:$|[_-])|"
    r"(?:^|[_-])pollen(?:$|[_-])|(?:^|[_-])no2(?:$|[_-])|water[_-]?treatment|"
    r"japanese[_-]?vowels|synthetic[_-]?control|forex|time[_-]?series|forecast|"
    r"electricity(?:_|$)|airlines(?:_|$)|ozone[_-]?level[_-]?8hr)",
    re.IGNORECASE,
)
IMAGE_TEXT_NAME_RE = re.compile(
    r"(?:optdigits|mfeat|authorship|splice|mnist|image|document|text)", re.IGNORECASE
)
ID_NAME_RE = re.compile(
    r"(?:^id$|^id[_-]|[_-]id$|[_-]id[_-]|id$|index|key)", re.IGNORECASE
)

FRI_RE = re.compile(r"^fri_c\d+_\d+_\d+$", re.IGNORECASE)
MONKS_RE = re.compile(r"^monks-problems-\d+$", re.IGNORECASE)
SEED_GRID_RE = re.compile(
    r"^(.+?)_seed_\d+_nrows_\d+_nclasses_\d+_ncols_\d+_stratify_(?:true|false)$",
    re.IGNORECASE,
)
NUMBER_GRID_RE = re.compile(r"^(.+?)_\d+_\d+$", re.IGNORECASE)

# Frozen manual same-source aliases, established from names and metadata before
# calculating any Phase 0B AUROC. The canonical label is the earliest source name.
MANUAL_ALIAS_GROUPS = {
    "heart-h": {"heart-h", "hungarian"},
    "heart-c": {"heart-c", "cleve", "cleveland"},
    "prnn_fglass": {"prnn-fglass", "glass"},
    "spambase": {"spambase", "spam"},
    "thyroid": {"sick", "hypothyroid"},
    "credit-approval": {"credit-approval", "australian"},
    "breast-cancer": {"breast-cancer", "breasttumor"},
    "boston": {"boston", "boston-corrected"},
    "machine_cpu": {"machine-cpu", "cpu"},
    "cpu_act": {"cpu-act", "cpu-small"},
    "bank-simulation": {"bank8fm", "bank32nh"},
    "puma-simulation": {"puma32h", "puma8nh"},
    "mfeat": {"mfeat-morphological", "mfeat-fourier", "mfeat-zernike", "mfeat-karhunen"},
    "kdd_ipums_la": {"kdd-ipums-la-97-small", "ipums-la-98-small", "ipums-la-99-small"},
    "ilpd": {"ilpd"},
    "thoracic-surgery": {"thoracic-surgery"},
}

CSV_COLUMNS = [
    "openml_id", "name", "version", "source_suite", "family", "family_size",
    "n", "p_raw", "p_used", "minority_rate", "n_dropped_missing_cols",
    "n_dropped_id_cols", "best_feature_name", "best_feature_fold_consistency",
    "auc_best_feature", "auc_lr", "SCR", "gap", "N_eff", "N_eff_norm",
    "excluded_reason",
]
INTERNAL_COLUMNS = CSV_COLUMNS + ["candidate_priority", "is_synthetic_family"]
STRATIFIERS = ["SCR", "gap", "N_eff", "N_eff_norm"]
PRIMARY_SELECTION_ORDER = ["N_eff", "N_eff_norm", "gap", "SCR"]


def normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    normalized = re.sub(r"^dataset-", "", normalized)
    normalized = re.sub(r"-(?:copy|reproduced)(?:-\d+)?$", "", normalized)
    normalized = re.sub(r"-(?:numeric|uci)$", "", normalized)
    return normalized


def build_family_annotations(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    normalized_names = {int(row.did): normalize_name(str(row.name)) for row in pool.itertuples()}
    alias_lookup: dict[str, str] = {}
    for canonical, aliases in MANUAL_ALIAS_GROUPS.items():
        for alias in aliases:
            alias_lookup[normalize_name(alias)] = canonical

    numbered_bases = pool["name"].astype(str).map(
        lambda name: NUMBER_GRID_RE.fullmatch(name).group(1).lower()
        if NUMBER_GRID_RE.fullmatch(name) else ""
    )
    numbered_counts = numbered_bases[numbered_bases.ne("")].value_counts()
    numbered_valid = set(numbered_counts[numbered_counts.ge(3)].index)

    raw_keys: dict[int, str] = {}
    synthetic: dict[int, bool] = {}
    for row in pool.itertuples():
        did = int(row.did)
        name = str(row.name)
        normalized = normalized_names[did]
        if FRI_RE.fullmatch(name):
            raw_keys[did] = "special:fri_c"
            synthetic[did] = True
        elif MONKS_RE.fullmatch(name):
            raw_keys[did] = "special:monks"
            synthetic[did] = True
        elif re.match(r"^GAMETES_", name, re.IGNORECASE):
            raw_keys[did] = "special:gametes"
            synthetic[did] = True
        elif re.match(r"^chscase_census\d+$", name, re.IGNORECASE):
            raw_keys[did] = "special:chscase_census"
            synthetic[did] = False
        else:
            seed_match = SEED_GRID_RE.fullmatch(name)
            number_match = NUMBER_GRID_RE.fullmatch(name)
            if seed_match:
                base = normalize_name(seed_match.group(1))
                base = alias_lookup.get(base, base)
                raw_keys[did] = f"source:{base}"
                synthetic[did] = True  # non-independent resampled grid
            elif number_match and number_match.group(1).lower() in numbered_valid:
                raw_keys[did] = f"grid:{number_match.group(1).lower()}"
                synthetic[did] = True
            else:
                base = alias_lookup.get(normalized, normalized)
                raw_keys[did] = f"source:{base}"
                synthetic[did] = False

    pool["_family_key"] = pool["did"].astype(int).map(raw_keys)
    pool["is_synthetic_family"] = pool["did"].astype(int).map(synthetic)

    canonical_by_key: dict[str, str] = {}
    for key, group in pool.sort_values(["version", "did"]).groupby("_family_key", sort=False):
        if key == "special:fri_c":
            canonical_by_key[key] = "fri_c"
        elif key == "special:monks":
            canonical_by_key[key] = "monks"
        elif key == "special:gametes":
            canonical_by_key[key] = "gametes"
        elif key == "special:chscase_census":
            canonical_by_key[key] = "chscase_census"
        elif key.startswith("grid:"):
            canonical_by_key[key] = key.split(":", 1)[1]
        else:
            real_rows = group.loc[~group["is_synthetic_family"]]
            canonical_by_key[key] = (
                str(real_rows.iloc[0]["name"])
                if len(real_rows)
                else key.split(":", 1)[1]
            )
    pool["family"] = pool["_family_key"].map(canonical_by_key)
    return pool.drop(columns=["_family_key"])


def structure_exclusion(row: dict[str, object]) -> str | None:
    name = str(row["name"])
    fmt = str(row.get("format", ""))
    p = int(float(row["NumberOfFeatures"]))
    if "sparse" in fmt.lower():
        return "excluded_high_dimensional_sparse_structure"
    if TIME_SERIES_NAME_RE.search(name):
        return "excluded_time_series_or_longitudinal_structure"
    if p >= 50 and IMAGE_TEXT_NAME_RE.search(name):
        return "excluded_image_or_text_derived_high_dimensional_table"
    return None


def make_unique_columns(columns: list[object]) -> list[str]:
    counts: Counter[str] = Counter()
    result: list[str] = []
    for value in columns:
        base = str(value)
        counts[base] += 1
        result.append(base if counts[base] == 1 else f"{base}__dup{counts[base]}")
    return result


def clean_and_screen_data(
    dataset: openml.datasets.OpenMLDataset,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object], dict[str, bool]]:
    target = dataset.default_target_attribute
    if not target:
        raise ValueError("missing_default_target_attribute")
    X, y, categorical_indicator, _ = dataset.get_data(target=target, dataset_format="dataframe")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("multiple_target_columns")
        y = y.iloc[:, 0]
    y = pd.Series(y)
    target_present = ~y.isna()
    X = X.loc[target_present].reset_index(drop=True)
    y = y.loc[target_present].reset_index(drop=True)
    if len(y) == 0:
        raise ValueError("all_target_values_missing")

    X.columns = make_unique_columns(list(X.columns))
    p_raw = X.shape[1]
    if categorical_indicator is None or len(categorical_indicator) != p_raw:
        categorical_map = {
            col: not pd.api.types.is_numeric_dtype(X[col].dtype) for col in X.columns
        }
    else:
        categorical_map = {
            col: bool(flag) for col, flag in zip(X.columns, categorical_indicator)
        }

    for col in X.columns:
        if not categorical_map[col]:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    missing_rate = X.isna().mean()
    missing_drop = list(missing_rate.index[missing_rate > MAX_COLUMN_MISSING_RATE])
    X = X.drop(columns=missing_drop)
    categorical_map = {c: categorical_map[c] for c in X.columns}

    id_drop: list[str] = []
    for col in X.columns:
        if ID_NAME_RE.search(col):
            id_drop.append(col)
        elif categorical_map[col] and X[col].nunique(dropna=True) >= 0.95 * len(X):
            id_drop.append(col)
    X = X.drop(columns=id_drop)
    categorical_map = {c: categorical_map[c] for c in X.columns}

    classes = y.astype("string").nunique(dropna=True)
    if classes != 2:
        raise ValueError(f"actual_number_of_classes={classes}")
    encoded_y = LabelEncoder().fit_transform(y.astype("string")).astype(int)
    counts = np.bincount(encoded_y)
    info = {
        "n": len(X),
        "p_raw": p_raw,
        "p_used": X.shape[1],
        "minority_rate": float(counts.min() / counts.sum()),
        "n_dropped_missing_cols": len(missing_drop),
        "n_dropped_id_cols": len(id_drop),
    }
    return X, encoded_y, info, categorical_map


def validate_actual_screen(info: dict[str, object]) -> str | None:
    if not (N_MIN <= int(info["n"]) <= N_MAX):
        return f"actual_n_outside_{N_MIN}_{N_MAX}"
    if not (P_MIN <= int(info["p_used"]) <= P_MAX):
        return f"p_used_outside_{P_MIN}_{P_MAX}"
    if not (MINORITY_RATE_MIN <= float(info["minority_rate"]) <= MINORITY_RATE_MAX):
        return "actual_minority_rate_outside_0.05_0.50"
    return None


def prepare_feature_frame(
    X: pd.DataFrame, categorical_map: dict[str, bool]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    X = X.copy()
    categorical_cols = [c for c in X.columns if categorical_map[c]]
    numeric_cols = [c for c in X.columns if not categorical_map[c]]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in categorical_cols:
        X[col] = X[col].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
    return X, numeric_cols, categorical_cols


def single_feature_percentile_scores(
    train: pd.Series, test: pd.Series, categorical: bool
) -> tuple[np.ndarray, np.ndarray]:
    if categorical:
        nonmissing = train.dropna().map(str)
        mode = nonmissing.mode(dropna=True)
        fill_value = str(mode.iloc[0]) if len(mode) else "__MISSING__"
        train_values = train.map(lambda value: fill_value if pd.isna(value) else str(value))
        test_values = test.map(lambda value: fill_value if pd.isna(value) else str(value))
        levels = sorted(train_values.unique().tolist())
        mapping = {level: i for i, level in enumerate(levels)}
        train_raw = train_values.map(mapping).astype(float).to_numpy()
        test_raw = test_values.map(mapping).fillna(-1).astype(float).to_numpy()
    else:
        train_num = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_num = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(train_num.median()) if train_num.notna().any() else 0.0
        train_raw = train_num.fillna(median).astype(float).to_numpy()
        test_raw = test_num.fillna(median).astype(float).to_numpy()
    ordered = np.sort(train_raw)
    train_scores = np.searchsorted(ordered, train_raw, side="right") / len(ordered)
    test_scores = np.searchsorted(ordered, test_raw, side="right") / len(ordered)
    return train_scores.astype(float), test_scores.astype(float)


def build_lr_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler", StandardScaler())]), numeric_cols)
        )
    if categorical_cols:
        transformers.append(
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
        )
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(
        C=1.0, l1_ratio=0.0, solver="lbfgs", max_iter=2000, random_state=SEED
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def calculate_metrics(
    X: pd.DataFrame, y: np.ndarray, categorical_map: dict[str, bool]
) -> tuple[dict[str, object], dict[str, float]]:
    X, numeric_cols, categorical_cols = prepare_feature_frame(X, categorical_map)
    features = list(X.columns)
    oof_features = np.full((len(X), len(features)), np.nan, dtype=float)
    oof_lr = np.full(len(X), np.nan, dtype=float)
    fold_winners: list[str] = []
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        corrected_train_aucs: list[float] = []
        for j, col in enumerate(features):
            train_scores, test_scores = single_feature_percentile_scores(
                X_train[col], X_test[col], categorical_map[col]
            )
            train_auc = float(roc_auc_score(y_train, train_scores))
            direction = 1 if train_auc >= 0.5 else -1
            corrected_train_aucs.append(train_auc if direction == 1 else 1.0 - train_auc)
            oof_features[test_idx, j] = test_scores if direction == 1 else 1.0 - test_scores
        fold_winners.append(features[int(np.argmax(corrected_train_aucs))])

        pipeline = build_lr_pipeline(numeric_cols, categorical_cols)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        oof_lr[test_idx] = pipeline.predict_proba(X_test)[:, 1]

    if np.isnan(oof_features).any() or np.isnan(oof_lr).any():
        raise RuntimeError("incomplete_out_of_fold_predictions")
    marginal = {
        feature: float(roc_auc_score(y, oof_features[:, j]))
        for j, feature in enumerate(features)
    }
    best_feature = max(features, key=lambda feature: marginal[feature])
    auc_best = float(marginal[best_feature])
    auc_lr = float(roc_auc_score(y, oof_lr))
    lifts = np.maximum(np.array(list(marginal.values()), dtype=float) - 0.5, 0.0)
    lift_sum = float(lifts.sum())
    lift_sq_sum = float(np.square(lifts).sum())
    n_eff = float(lift_sum**2 / lift_sq_sum) if lift_sq_sum > 0 else np.nan
    n_eff_norm = float(n_eff / len(features)) if np.isfinite(n_eff) else np.nan
    metrics = {
        "best_feature_name": best_feature,
        "best_feature_fold_consistency": int(sum(winner == best_feature for winner in fold_winners)),
        "auc_best_feature": auc_best,
        "auc_lr": auc_lr,
        "SCR": float((auc_best - 0.5) / (auc_lr - 0.5)) if auc_lr > NO_SIGNAL_AUC_LR_MAX else np.nan,
        "gap": float(auc_lr - auc_best),
        "N_eff": n_eff,
        "N_eff_norm": n_eff_norm,
        "excluded_reason": "" if auc_lr > NO_SIGNAL_AUC_LR_MAX else "auc_lr_le_0.52_no_usable_signal",
    }
    return metrics, marginal


def blank_result(row: dict[str, object], reason: str) -> dict[str, object]:
    return {
        "openml_id": int(row["did"]),
        "name": str(row["name"]),
        "version": int(float(row["version"])),
        "source_suite": str(row["source_suite"]),
        "family": str(row["family"]),
        "family_size": 0,
        "n": int(float(row["NumberOfInstances"])),
        "p_raw": int(float(row["NumberOfFeatures"])),
        "p_used": np.nan,
        "minority_rate": float(row["minority_rate_meta"]),
        "n_dropped_missing_cols": np.nan,
        "n_dropped_id_cols": np.nan,
        "best_feature_name": "",
        "best_feature_fold_consistency": np.nan,
        "auc_best_feature": np.nan,
        "auc_lr": np.nan,
        "SCR": np.nan,
        "gap": np.nan,
        "N_eff": np.nan,
        "N_eff_norm": np.nan,
        "excluded_reason": reason,
        "candidate_priority": int(row["candidate_priority"]),
        "is_synthetic_family": bool(row["is_synthetic_family"]),
    }


def worker(row: dict[str, object]) -> tuple[dict[str, object], dict[str, float] | None, str | None]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    openml.config.cache_directory = str(OPENML_CACHE)
    did = int(row["did"])
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            dataset = openml.datasets.get_dataset(
                did, download_data=True, download_qualities=True, download_features_meta_data=True
            )
            X, y, info, categorical_map = clean_and_screen_data(dataset)
            actual_reason = validate_actual_screen(info)
            result = blank_result(row, actual_reason or "")
            result.update(info)
            if actual_reason:
                return result, None, None
            metrics, marginal = calculate_metrics(X, y, categorical_map)
            result.update(metrics)
            return result, marginal if not metrics["excluded_reason"] else None, None
        except Exception as exc:
            last_error = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2**attempt)
    assert last_error is not None
    reason = re.sub(r"\s+", " ", f"processing_failure:{type(last_error).__name__}:{last_error}").strip()
    return blank_result(row, reason), None, traceback.format_exc()


def checkpoint_results(rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).reindex(columns=INTERNAL_COLUMNS).sort_values("candidate_priority").to_csv(
        CHECKPOINT, index=False
    )


def run_candidates(pool: pd.DataFrame) -> pd.DataFrame:
    MARGINAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT.exists():
        prior = pd.read_csv(CHECKPOINT)
        rows = prior.to_dict(orient="records")
        processed = set(pd.to_numeric(prior["openml_id"]).astype(int))
        print(f"RESUME processed={len(rows)}", flush=True)
    else:
        rows = []
        processed: set[int] = set()

    pending: list[dict[str, object]] = []
    for row in pool.to_dict(orient="records"):
        did = int(row["did"])
        if did in processed:
            continue
        reason = structure_exclusion(row)
        if reason:
            rows.append(blank_result(row, reason))
            print(f"{did}\t{row['source_suite']}\tEXCLUDED\t{reason}", flush=True)
        else:
            pending.append(row)
    checkpoint_results(rows)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, row): row for row in pending}
        for future in as_completed(futures):
            source_row = futures[future]
            did = int(source_row["did"])
            try:
                result, marginal, error_text = future.result()
            except Exception as exc:
                reason = re.sub(r"\s+", " ", f"worker_failure:{type(exc).__name__}:{exc}").strip()
                result = blank_result(source_row, reason)
                marginal = None
                error_text = traceback.format_exc()
            rows.append(result)
            if marginal is not None:
                (MARGINAL_CHECKPOINT_DIR / f"{did}.json").write_text(
                    json.dumps(marginal, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            if error_text:
                (WORK / f"phase0b_error_{did}.txt").write_text(error_text, encoding="utf-8")
            status = "INCLUDED" if not result["excluded_reason"] else "EXCLUDED"
            print(
                f"{did}\t{result['source_suite']}\t{status}\t{result['excluded_reason'] or 'included'}",
                flush=True,
            )
            checkpoint_results(rows)

    results = pd.DataFrame(rows).reindex(columns=INTERNAL_COLUMNS).sort_values("candidate_priority")
    if len(results) != len(pool):
        raise AssertionError(f"processed_rows={len(results)} candidate_rows={len(pool)}")
    return results


def load_included_marginals(results: pd.DataFrame) -> dict[str, dict[str, float]]:
    marginal: dict[str, dict[str, float]] = {}
    included = results["excluded_reason"].fillna("").eq("")
    for did in results.loc[included, "openml_id"].astype(int):
        path = MARGINAL_CHECKPOINT_DIR / f"{did}.json"
        if not path.exists():
            raise AssertionError(f"missing_marginal_checkpoint_{did}")
        marginal[str(did)] = json.loads(path.read_text(encoding="utf-8"))
    return marginal


def neff_from_values(values: list[float]) -> float:
    lifts = np.maximum(np.asarray(values, dtype=float) - 0.5, 0.0)
    denom = float(np.square(lifts).sum())
    return float(lifts.sum() ** 2 / denom) if denom > 0 else np.nan


def run_self_checks(
    results: pd.DataFrame, marginal: dict[str, dict[str, float]]
) -> dict[str, object]:
    included = results.loc[results["excluded_reason"].fillna("").eq("")].copy()
    fri_sizes = included.loc[included["family"].eq("fri_c"), "family_size"]
    assert len(fri_sizes) > 1 and fri_sizes.max() > 1
    for row in included.itertuples():
        aucs = marginal[str(int(row.openml_id))]
        assert len(aucs) == int(row.p_used)
        assert abs(max(aucs.values()) - float(row.auc_best_feature)) <= 1e-9

    sample = included.sample(n=3, random_state=SEED)
    sample_checks: list[dict[str, object]] = []
    for row in sample.itertuples():
        aucs = marginal[str(int(row.openml_id))]
        manual = neff_from_values(list(aucs.values()))
        assert (np.isnan(manual) and np.isnan(row.N_eff)) or abs(manual - float(row.N_eff)) <= 1e-12
        positive_count = sum(max(value - 0.5, 0.0) > 0 for value in aucs.values())
        if positive_count:
            equal_lifts = [0.6] * positive_count
            equal_neff = neff_from_values(equal_lifts)
            assert abs(equal_neff - positive_count) <= 1e-12
        else:
            equal_neff = np.nan
        sample_checks.append(
            {
                "openml_id": int(row.openml_id),
                "name": row.name,
                "stored_N_eff": float(row.N_eff),
                "manual_N_eff": manual,
                "positive_lift_feature_count": positive_count,
                "equal_lift_N_eff": equal_neff,
            }
        )

    script_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    model_calls = {
        node.func.id
        for node in ast.walk(script_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and ("Classifier" in node.func.id or "Regressor" in node.func.id or node.func.id == "LogisticRegression")
    }
    assert model_calls == {"LogisticRegression"}
    check = {
        "fri_c_family_size": int(fri_sizes.max()),
        "sample_N_eff_checks": sample_checks,
        "auc_best_matches_marginal_max_for_all_included": True,
        "marginal_feature_count_matches_p_used_for_all_included": True,
        "model_call_audit": sorted(model_calls),
        "stratifiers_clipped": False,
    }
    SELF_CHECK_PATH.write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return check


def descriptive(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    q = clean.quantile([0.25, 0.5, 0.75])
    return {
        "min": float(clean.min()), "Q1": float(q.loc[0.25]),
        "median": float(q.loc[0.5]), "Q3": float(q.loc[0.75]), "max": float(clean.max()),
    }


def fmt_stats(stats: dict[str, float]) -> str:
    return " / ".join(f"{key}={value:.4g}" for key, value in stats.items())


def correlation_table(included: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    correlations: dict[str, dict[str, tuple[float, float]]] = {}
    for stratifier in STRATIFIERS:
        correlations[stratifier] = {}
        for covariate in ["auc_lr", "n", "p_used", "minority_rate"]:
            rho, p_value = spearmanr(
                pd.to_numeric(included[stratifier], errors="coerce"),
                pd.to_numeric(included[covariate], errors="coerce"),
                nan_policy="omit",
            )
            correlations[stratifier][covariate] = (float(rho), float(p_value))
    return correlations


def main_stratifier_eligibility(
    included: pd.DataFrame, correlations: dict[str, dict[str, tuple[float, float]]]
) -> dict[str, dict[str, object]]:
    decisions: dict[str, dict[str, object]] = {}
    for stratifier in STRATIFIERS:
        values = pd.to_numeric(included[stratifier], errors="coerce").dropna()
        bounded = bool((values >= 0).all() and (values <= 2).all())
        rho = correlations[stratifier]["auc_lr"][0]
        decisions[stratifier] = {
            "abs_rho_auc_lr": abs(rho),
            "bounded_0_to_2": bounded,
            "eligible": bool(abs(rho) <= 0.25 and bounded),
        }
    return decisions


def tertile_rows(target: pd.DataFrame, included: pd.DataFrame, stratifier: str) -> list[dict[str, object]]:
    all_values = pd.to_numeric(included[stratifier], errors="coerce")
    q1, q2 = all_values.quantile([1 / 3, 2 / 3]).tolist()
    target_values = pd.to_numeric(target[stratifier], errors="coerce")
    masks = [
        target_values <= q1,
        (target_values > q1) & (target_values <= q2),
        target_values > q2,
    ]
    labels = ["low", "middle", "high"]
    rows: list[dict[str, object]] = []
    for label, mask in zip(labels, masks):
        subset = target.loc[mask.fillna(False)]
        rows.append(
            {
                "stratum": label,
                "dataset_count": len(subset),
                "family_count": subset["family"].nunique(),
            }
        )
    undefined = target.loc[target_values.isna()]
    if len(undefined):
        rows.append(
            {"stratum": "undefined", "dataset_count": len(undefined),
             "family_count": undefined["family"].nunique()}
        )
    return [{"q1": q1, "q2": q2, **row} for row in rows]


def save_histograms(included: pd.DataFrame, path: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    colors = {False: "#4F8F8B", True: "#D48A55"}
    labels = {False: "Independent/real source", True: "Synthetic/resampled family"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.90, wspace=0.28, hspace=0.38)
    titles = {
        "SCR": "Signal concentration ratio",
        "gap": "Multivariable increment (gap)",
        "N_eff": "Effective contributing features",
        "N_eff_norm": "Normalized effective features",
    }
    for panel, (ax, stratifier) in enumerate(zip(axes.flat, STRATIFIERS)):
        all_values = pd.to_numeric(included[stratifier], errors="coerce").dropna()
        bins = np.histogram_bin_edges(all_values, bins=max(12, min(24, int(np.ceil(np.sqrt(len(all_values)))))))
        for synthetic in [False, True]:
            values = pd.to_numeric(
                included.loc[included["is_synthetic_family"].eq(synthetic), stratifier], errors="coerce"
            ).dropna()
            ax.hist(
                values, bins=bins, color=colors[synthetic], alpha=0.72,
                edgecolor="white", linewidth=0.5, label=labels[synthetic]
            )
        ax.set_title(titles[stratifier], loc="left", fontweight="bold", fontsize=8)
        ax.set_xlabel(stratifier)
        ax.set_ylabel("Datasets")
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.text(-0.12, 1.05, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 0.985))
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def generate_outputs(
    results: pd.DataFrame,
    marginal: dict[str, dict[str, float]],
    self_checks: dict[str, object],
) -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    included_mask = results["excluded_reason"].fillna("").eq("")
    family_counts = results.loc[included_mask, "family"].value_counts()
    results = results.copy()
    results["family_size"] = results["family"].map(family_counts).fillna(0).astype(int)
    public = results.reindex(columns=CSV_COLUMNS)
    public.to_csv(OUTPUT / "datasets_stratifiers.csv", index=False, float_format="%.12g")
    (OUTPUT / "marginal_aucs.json").write_text(
        json.dumps(marginal, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    run_lines: list[str] = []
    for row in results.itertuples():
        reason = "" if pd.isna(row.excluded_reason) else str(row.excluded_reason)
        included = reason == ""
        run_lines.append(
            f"{int(row.openml_id)}\t{row.source_suite}\t"
            f"{'included' if included else 'excluded'}\t{'included' if included else reason}"
        )
    (OUTPUT / "run_log.txt").write_text("\n".join(run_lines) + "\n", encoding="utf-8")

    included = results.loc[included_mask].copy()
    excluded = results.loc[~included_mask].copy()
    target = included.loc[included["auc_lr"].between(TARGET_REGION_MIN, TARGET_REGION_MAX, inclusive="both")]
    target_family_count = target["family"].nunique()
    target_real_family_count = target.loc[~target["is_synthetic_family"], "family"].nunique()
    independent_real_family_count = included.loc[~included["is_synthetic_family"], "family"].nunique()
    correlations = correlation_table(included)
    eligibility = main_stratifier_eligibility(included, correlations)
    eligible_variables = [name for name in PRIMARY_SELECTION_ORDER if eligibility[name]["eligible"]]
    selected_main = eligible_variables[0] if eligible_variables else None

    if target_family_count >= 30:
        phase_decision = "通过，Phase 1 可启动"
    elif target_family_count >= 15:
        phase_decision = "部分通过，需继续扩池"
    else:
        phase_decision = "不通过，目标区域独立家族不足 15 个"
    stratifier_decision = (
        f"{selected_main} 可作为主分层变量" if selected_main
        else "警告：四个变量均不满足主分层变量标准，需另行设计"
    )

    save_histograms(included, OUTPUT / "stratifier_hists.png")

    report: list[str] = [f"# 结论：{phase_decision}；{stratifier_decision}。", ""]
    if independent_real_family_count < 150:
        report += [
            f"> **扩池目标警告：** 纳入的独立真实来源家族为 {independent_real_family_count}，低于目标 150。",
            "",
        ]
    report += [
        "## 数据集流转与家族结构", "",
        f"- 冻结候选数：{len(results)}",
        f"- 纳入统计：{len(included)}",
        f"- 排除或失败：{len(excluded)}",
        f"- 纳入后的全部独立来源家族数：{included['family'].nunique()}",
        f"- 纳入后的独立真实来源家族数（排除合成/重采样网格）：{independent_real_family_count}",
        f"- 合成/重采样家族数：{included.loc[included['is_synthetic_family'], 'family'].nunique()}",
        "- 类型计数可重叠：同一真实来源家族可同时含原始版本与其 seed 重采样版本。",
        "", "### 排除原因", "", "| 原因 | 数量 |", "|---|---:|",
    ]
    for reason, count in excluded["excluded_reason"].value_counts().sort_index().items():
        report.append(f"| {reason} | {count} |")
    report += ["", "### 最大的五个家族", "", "| 家族 | 纳入成员数 | 类型 |", "|---|---:|---|"]
    for family, count in family_counts.head(5).items():
        family_types = included.loc[included["family"].eq(family), "is_synthetic_family"].astype(bool)
        if family_types.all():
            family_type = "合成/重采样"
        elif family_types.any():
            family_type = "真实来源（含重采样版本）"
        else:
            family_type = "真实来源"
        report.append(f"| {family} | {count} | {family_type} |")

    report += ["", "## 四种候选分层变量", "", "| 变量 | min | Q1 | median | Q3 | max | 未定义数 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for stratifier in STRATIFIERS:
        stats = descriptive(included[stratifier])
        undefined = pd.to_numeric(included[stratifier], errors="coerce").isna().sum()
        report.append(
            f"| {stratifier} | {stats['min']:.4g} | {stats['Q1']:.4g} | {stats['median']:.4g} | "
            f"{stats['Q3']:.4g} | {stats['max']:.4g} | {undefined} |"
        )

    report += ["", "## 稳定性比较", "",
               "单元格为 Spearman ρ（双侧 p 值）。与 `auc_lr` 的 |ρ| 越小，说明受整体可预测性污染越弱。", "",
               "| 变量 | auc_lr | n | p_used | minority_rate | [0,2] 有界 | 主变量标准 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for stratifier in STRATIFIERS:
        cells = []
        for covariate in ["auc_lr", "n", "p_used", "minority_rate"]:
            rho, p_value = correlations[stratifier][covariate]
            cells.append(f"{rho:.3f} ({p_value:.3g})")
        decision = eligibility[stratifier]
        report.append(
            f"| {stratifier} | {' | '.join(cells)} | "
            f"{'是' if decision['bounded_0_to_2'] else '否'} | {'满足' if decision['eligible'] else '不满足'} |"
        )

    report += ["", "## 目标区域", "",
               f"目标区域定义为 `auc_lr ∈ [{TARGET_REGION_MIN:.2f}, {TARGET_REGION_MAX:.2f}]`。",
               "", f"- 数据集数：{len(target)}",
               f"- 独立来源家族数（判定使用）：{target_family_count}",
               f"- 其中独立真实来源家族数：{target_real_family_count}", ""]
    for stratifier in STRATIFIERS:
        rows = tertile_rows(target, included, stratifier)
        q1, q2 = rows[0]["q1"], rows[0]["q2"]
        report += [f"### {stratifier} 三分位（全纳入集切点：{q1:.4g}, {q2:.4g}）", "",
                   "| 层 | 目标区数据集数 | 目标区独立家族数 |", "|---|---:|---:|"]
        for item in rows:
            report.append(f"| {item['stratum']} | {item['dataset_count']} | {item['family_count']} |")
        report.append("")

    report += ["## 分布图", "", "![Four stratifier distributions](stratifier_hists.png)", "",
               "## 自检", "",
               f"- `fri_c` 在纳入集中的家族大小：{self_checks['fri_c_family_size']}（家族标注已生效）。",
               "- 所有纳入数据集在 `marginal_aucs.json` 中的特征数均等于 `p_used`。",
               "- 所有 `auc_best_feature` 均与对应边际 AUROC 最大值在 1e-9 内一致。",
               "- 模型调用审计仅发现 `LogisticRegression`。",
               "- 四种分层变量均按定义原样计算，没有结果后裁剪。", "",
               "### 随机 3 个数据集的 N_eff 手算", "",
               "| OpenML ID | 数据集 | 脚本 N_eff | 手算 N_eff | 正 lift 特征数 | 等 lift 检查 |",
               "|---:|---|---:|---:|---:|---:|"]
    for item in self_checks["sample_N_eff_checks"]:
        report.append(
            f"| {item['openml_id']} | {item['name']} | {item['stored_N_eff']:.8g} | "
            f"{item['manual_N_eff']:.8g} | {item['positive_lift_feature_count']} | "
            f"{item['equal_lift_N_eff']:.8g} |"
        )

    report += ["", "## 方法与预冻结选择", "",
               "- 候选来源优先级为 CC18、AMLB classification-all（suite 271）、OpenML 全库按 ID 升序；元数据合格且同名版本去重后冻结前 500 个候选，因已达到目标未补充 UCI。",
               "- 外层验证为 `StratifiedKFold(5, shuffle=True, random_state=42)`。每个特征分别在训练折决定方向，并用训练折经验百分位映射产生 held-out 分值；全部特征的五折 held-out 分值汇总后计算边际 AUROC。",
               "- 类别特征使用训练折内排序类别编码；数值特征的单变量缺失用训练折中位数处理。逻辑回归预处理为数值中位数插补与标准化、类别众数插补与 one-hot。",
               "- `lift_j=max(auc_j-0.5,0)` 仅是 N_eff 定义的一部分；SCR、gap、N_eff 与 N_eff_norm 均未裁剪。",
               "- 本阶段没有运行任何模型对比；唯一拟合的模型是用于参照 AUROC 的 L2 逻辑回归。", ""]
    (OUTPUT / "stratifier_report.md").write_text("\n".join(report), encoding="utf-8")

    environment = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "openml_version": openml.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": mpl.__version__,
        "random_seed": SEED,
        "parallel_workers": MAX_WORKERS,
        "openml_suite_ids": {"CC18": 99, "AMLB_classification_all": 271},
        "candidate_rules": {
            "target_count": 500,
            "priority": ["CC18", "AMLB", "openml-scan", "UCI_optional"],
            "same_name_dedupe_key": ["name", "n", "p"],
            "same_name_dedupe_keep": "minimum_version_then_minimum_openml_id",
        },
        "screening_rules": {
            "task_type": "binary_classification", "n_min": N_MIN, "n_max": N_MAX,
            "p_raw_min": P_MIN, "p_raw_max": P_MAX, "p_used_min": P_MIN,
            "p_used_max": P_MAX, "minority_rate_min": MINORITY_RATE_MIN,
            "minority_rate_max": MINORITY_RATE_MAX,
            "max_column_missing_rate": MAX_COLUMN_MISSING_RATE,
            "id_name_regex": ID_NAME_RE.pattern,
            "non_numeric_id_unique_rate_min": 0.95,
            "time_series_name_regex": TIME_SERIES_NAME_RE.pattern,
            "image_text_name_regex": IMAGE_TEXT_NAME_RE.pattern,
            "image_text_high_dimensional_p_min": 50,
            "sparse_format_excluded": True,
            "no_signal_if_auc_lr_le": NO_SIGNAL_AUC_LR_MAX,
        },
        "family_rules": {
            "fri_regex": FRI_RE.pattern, "monks_regex": MONKS_RE.pattern,
            "seed_grid_regex": SEED_GRID_RE.pattern, "number_grid_regex": NUMBER_GRID_RE.pattern,
            "number_grid_min_pool_occurrences": 3,
            "manual_alias_groups": {key: sorted(value) for key, value in MANUAL_ALIAS_GROUPS.items()},
            "family_size_scope": "included_datasets_only",
        },
        "outer_cv": {"class": "StratifiedKFold", "n_splits": N_SPLITS,
                     "shuffle": True, "random_state": SEED},
        "stratifier_rules": {
            "target_region_auc_lr": [TARGET_REGION_MIN, TARGET_REGION_MAX],
            "primary_eligibility": "abs(Spearman rho with auc_lr) <= 0.25 and all finite values in [0,2]",
            "primary_selection_order": PRIMARY_SELECTION_ORDER,
            "clipped": False,
        },
        "forbidden_model_comparison_run": False,
    }
    (OUTPUT / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with zipfile.ZipFile(OUTPUT_PARENT / "phase0b_outputs.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.iterdir()):
            archive.write(path, arcname=f"phase0b_outputs/{path.name}")


def validate_outputs(results: pd.DataFrame, marginal: dict[str, dict[str, float]]) -> None:
    data = pd.read_csv(OUTPUT / "datasets_stratifiers.csv")
    included = data["excluded_reason"].isna() | data["excluded_reason"].eq("")
    assert len(data) == 500
    assert data.columns.tolist() == CSV_COLUMNS
    assert data.loc[included, ["auc_best_feature", "auc_lr", "SCR", "gap"]].notna().all().all()
    assert (data.loc[included, "auc_lr"] > NO_SIGNAL_AUC_LR_MAX).all()
    assert set(map(str, data.loc[included, "openml_id"].astype(int))) == set(marginal)
    assert data.loc[data["family"].eq("fri_c") & included, "family_size"].max() > 1
    assert (OUTPUT / "stratifier_report.md").read_text(encoding="utf-8").startswith("# 结论：")
    assert len((OUTPUT / "run_log.txt").read_text(encoding="utf-8").splitlines()) == 500
    with zipfile.ZipFile(OUTPUT_PARENT / "phase0b_outputs.zip") as archive:
        assert archive.testzip() is None


def main() -> None:
    openml.config.cache_directory = str(OPENML_CACHE)
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(CANDIDATES)
    if len(pool) != 500:
        raise AssertionError(f"candidate_pool_size={len(pool)}")
    pool = build_family_annotations(pool)
    family_preview = pool.groupby(["family", "is_synthetic_family"]).size().sort_values(ascending=False)
    family_preview.to_csv(WORK / "phase0b_frozen_family_preview.csv", header=["candidate_count"])
    results = run_candidates(pool)
    marginal = load_included_marginals(results)
    included_mask = results["excluded_reason"].fillna("").eq("")
    family_counts = results.loc[included_mask, "family"].value_counts()
    results["family_size"] = results["family"].map(family_counts).fillna(0).astype(int)
    checks = run_self_checks(results, marginal)
    generate_outputs(results, marginal, checks)
    validate_outputs(results, marginal)
    print(f"FINAL_INCLUDED={int(included_mask.sum())}", flush=True)
    print(f"FINAL_EXCLUDED={int((~included_mask).sum())}", flush=True)
    print(f"OUTPUT={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
