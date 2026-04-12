"""
Shared utilities for the Saint-Gobain Distribution (SGDB) forecasting pipeline.

This module is imported by every notebook in the pipeline. Keep it:
- pure-Python where possible
- stateless (no implicit globals)
- Databricks-friendly (only depends on pyspark + numpy + pandas)

The functions here handle the two concerns that recur everywhere:
  1. Parsing and arithmetic on the "YYYY-WW" `semaine` string.
  2. WAPE computation (the competition metric).
  3. Spark window helpers that guarantee no target leakage.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F


# ----------------------------------------------------------------------------
# 1. SEMAINE PARSING
# ----------------------------------------------------------------------------
# The raw tables store the week as "YYYY-WW" (e.g. "2024-07").
# We materialise two helper columns on any DataFrame we touch:
#   - annee   (int)
#   - num_sem (int)
#   - week_id (int = annee * 100 + num_sem, monotonically sortable)
# Sorting by week_id gives the correct chronological order, which we rely on
# for every window function downstream.

def add_time_columns(df: DataFrame, semaine_col: str = "semaine") -> DataFrame:
    """Parse `YYYY-WW` strings into annee, num_sem and a sortable week_id."""
    return (
        df.withColumn("annee", F.split(F.col(semaine_col), "-").getItem(0).cast("int"))
          .withColumn("num_sem", F.split(F.col(semaine_col), "-").getItem(1).cast("int"))
          .withColumn("week_id", F.col("annee") * F.lit(100) + F.col("num_sem"))
    )


def semaine_minus_years(semaine_col: Column, years: int) -> Column:
    """Return the `YYYY-WW` string `years` years before `semaine_col`."""
    annee = F.split(semaine_col, "-").getItem(0).cast("int") - F.lit(years)
    num = F.split(semaine_col, "-").getItem(1)
    return F.concat(annee.cast("string"), F.lit("-"), num)


# ----------------------------------------------------------------------------
# 2. WAPE METRIC
# ----------------------------------------------------------------------------
# WAPE = sum(|actual - predicted|) / sum(|actual|)
# Low is good. The competition uses this exact formulation with a tiny epsilon
# to avoid division by zero when a slice has zero actuals.

EPS = 1e-10


def wape_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Numpy WAPE — used inside LightGBM feval and evaluation notebooks."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPS))


def wape_pandas(df: pd.DataFrame, y_col: str, pred_col: str) -> float:
    return wape_numpy(df[y_col].to_numpy(), df[pred_col].to_numpy())


def wape_spark(df: DataFrame, y_col: str = "quantite", pred_col: str = "prediction") -> float:
    """Compute WAPE on a Spark DataFrame and return a Python float."""
    row = df.agg(
        (
            F.sum(F.abs(F.col(y_col).cast("double") - F.col(pred_col).cast("double")))
            / (F.sum(F.abs(F.col(y_col).cast("double"))) + F.lit(EPS))
        ).alias("wape")
    ).collect()[0]
    return float(row["wape"])


def wape_lgb_feval(y_pred, dataset):
    """LightGBM custom metric — lower is better, so third return is False."""
    y_true = dataset.get_label()
    return "wape", wape_numpy(y_true, y_pred), False


# ----------------------------------------------------------------------------
# 3. SPARK WINDOW HELPERS (NO LEAKAGE)
# ----------------------------------------------------------------------------
# Every rolling/expanding feature we engineer must:
#   - be partitioned by (code_agence, code_article),
#   - be ordered by week_id,
#   - and end strictly *before* the current row (upper bound -1).
#
# These helpers centralise that contract so we cannot accidentally include the
# current row in a feature window — a silent leak would wreck validation.

PAIR_KEYS = ["code_agence", "code_article"]


def pair_window_lookback(n: int) -> Window:
    """Rolling window of the last `n` rows, excluding the current row."""
    return (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(-n, -1)
    )


def pair_window_expanding() -> Window:
    """Expanding window over all prior rows, excluding the current row."""
    return (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )


def agency_window_expanding() -> Window:
    return (
        Window.partitionBy("code_agence")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )


def article_window_expanding() -> Window:
    return (
        Window.partitionBy("code_article")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )


def seasonal_window_expanding() -> Window:
    """Same-week-of-year history, excluding the current row.

    We partition by (pair, num_sem) and order by annee so we only see past
    occurrences of the same calendar week. This is how we derive seasonal
    stats without leaking the current row's value.
    """
    return (
        Window.partitionBy(*PAIR_KEYS, "num_sem")
        .orderBy("annee")
        .rowsBetween(Window.unboundedPreceding, -1)
    )


def lag(col: str, n: int) -> Column:
    return F.lag(F.col(col), n).over(
        Window.partitionBy(*PAIR_KEYS).orderBy("week_id")
    )


# ----------------------------------------------------------------------------
# 4. CATEGORICAL ENCODING
# ----------------------------------------------------------------------------
# We label-encode categoricals into dense integers on the driver (14 agencies,
# ~162 families — all fit in memory). Using StringIndexer would force another
# Spark shuffle per column and would not guarantee the same integer mapping
# between training and inference, which matters because LightGBM's categorical
# feature handling uses those integers directly.

def collect_label_encoder(df: DataFrame, col: str) -> dict:
    """Build a deterministic value -> int mapping sorted by raw value."""
    values = [row[col] for row in df.select(col).distinct().orderBy(col).collect()]
    return {v: i for i, v in enumerate(values)}


def apply_label_encoder(df: DataFrame, col: str, mapping: dict, default: int = -1) -> DataFrame:
    """Add `{col}_enc` to a Spark DataFrame given a pre-computed mapping."""
    if not mapping:
        return df.withColumn(f"{col}_enc", F.lit(default))
    # Build a CASE WHEN expression. For 14 agencies or ~162 families this
    # stays small enough to remain a cheap pushdown-friendly expression.
    expr = F.lit(default)
    for value, idx in mapping.items():
        expr = F.when(F.col(col) == F.lit(value), F.lit(idx)).otherwise(expr)
    return df.withColumn(f"{col}_enc", expr)


# ----------------------------------------------------------------------------
# 5. MISC
# ----------------------------------------------------------------------------

def drop_columns_if_exist(df: DataFrame, cols: Iterable[str]) -> DataFrame:
    present = set(df.columns)
    for c in cols:
        if c in present:
            df = df.drop(c)
    return df


def nonneg_round(col: Column) -> Column:
    """Clip at zero and round to the nearest non-negative integer."""
    return F.greatest(F.round(col), F.lit(0.0)).cast("long")


# ----------------------------------------------------------------------------
# 6. PANDAS-LEVEL FEATURE PREPARATION (clipping + derived features)
# ----------------------------------------------------------------------------
# Called once on each DataFrame (train, val, test, inference) right after
# toPandas() and before build_xy(). Keeps all clipping / derived-feature
# logic in a single place so train and inference stay aligned.

# Columns that are ratios (output of division) — clip to [-10, 10].
_RATIO_COLS = [
    "ratio_lag52_vs_pair_mean_lag26",
    "ratio_band26_52_vs_pair_mean_lag26",
    "trend_band_26_52_vs_52_104",
    "yoy_ratio",
]

# Columns that are coefficients of variation — clip to [0, 5].
_CV_COLS = [
    "pair_cv_lag26",
    "band_cv_26_52", "band_cv_27_52", "band_cv_39_65",
    "band_cv_52_104", "band_cv_78_104", "band_cv_104_156",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip dangerous features and add derived columns (in-place on a copy).

    This function is the single source of truth for post-load feature
    transforms.  Call it on every pandas DataFrame **before** ``build_xy``.

    Changes applied:
    1. Clip ratio features to [-10, 10] to prevent extreme-split overfit.
    2. Clip CV features to [0, 5].
    3. Clip ``pair_last_nonzero_gap`` to [0, 104].
    4. Derive ``detrended_lag52`` ��� trend-corrected year-over-year lag.
    5. Derive ``demand_profile`` — Syntetos-Boylan 4-class categorisation.
    """
    df = df.copy()

    # --- 1. Clip ratios -------------------------------------------------------
    for c in _RATIO_COLS:
        if c in df.columns:
            df[c] = df[c].clip(-10.0, 10.0)

    # --- 2. Clip CVs ----------------------------------------------------------
    for c in _CV_COLS:
        if c in df.columns:
            df[c] = df[c].clip(0.0, 5.0)

    # --- 3. Clip pair_last_nonzero_gap ----------------------------------------
    if "pair_last_nonzero_gap" in df.columns:
        df["pair_last_nonzero_gap"] = df["pair_last_nonzero_gap"].clip(0, 104)

    # --- 4. detrended_lag52 ---------------------------------------------------
    # lag_52 corrected for the local trend: what last year's same-week value
    # would be if the trend were flat.  Uses pair_mean_lag26 (recent level) /
    # band_nonzero_mean_52_104 (older level) as the trend ratio.
    if {"lag_52", "pair_mean_lag26", "band_nonzero_mean_52_104"}.issubset(df.columns):
        denom = df["band_nonzero_mean_52_104"].replace(0.0, np.nan)
        trend_ratio = (df["pair_mean_lag26"] / denom).clip(0.1, 10.0)
        df["detrended_lag52"] = (df["lag_52"] * trend_ratio).clip(0.0, None)
        df["detrended_lag52"] = df["detrended_lag52"].fillna(df["lag_52"])
    elif "lag_52" in df.columns:
        # Fallback: no trend correction possible, keep lag_52 as-is.
        df["detrended_lag52"] = df["lag_52"]

    # --- 5. demand_profile (Syntetos-Boylan 4-class) --------------------------
    # 0 = SMOOTH        (frequent + regular volume)
    # 1 = ERRATIC       (frequent + irregular volume)
    # 2 = INTERMITTENT  (rare + regular volume)
    # 3 = LUMPY         (rare + irregular volume)
    # ADI threshold 1.32  →  zero_rate threshold = 1 - 1/1.32 ≈ 0.24
    # CV² threshold 0.49  →  CV threshold ≈ 0.7
    if {"pair_zero_rate_lag26", "pair_cv_lag26"}.issubset(df.columns):
        zr = df["pair_zero_rate_lag26"].fillna(1.0)
        cv = df["pair_cv_lag26"].fillna(0.0)
        profile = np.where(
            zr < 0.24,
            np.where(cv < 0.7, 0, 1),    # SMOOTH / ERRATIC
            np.where(cv < 0.7, 2, 3),    # INTERMITTENT / LUMPY
        )
        df["demand_profile"] = profile.astype(np.int32)
    else:
        df["demand_profile"] = np.int32(3)  # default to LUMPY if inputs missing

    return df
