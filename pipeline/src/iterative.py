"""Fast pandas/numpy iterative scoring helpers.

The gold tables deliberately mask future targets inside validation/test/final
horizons. These helpers rebuild target-derived features week by week and feed
the model's own predictions back into the state before scoring the next week.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 4, 8, 13, 26, 52, 104)
DEFAULT_ROLLING_WINDOWS = (4, 8, 13, 26, 52)
DEFAULT_ROLLING_MEDIAN_WINDOWS = (4, 13)


def ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with annee, num_sem and week_id available."""
    out = df.copy()
    if "week_id" not in out.columns:
        if "semaine" not in out.columns:
            raise ValueError("Need either week_id or semaine to build time columns.")
        parts = out["semaine"].astype(str).str.split("-", expand=True)
        out["annee"] = parts[0].astype("int32")
        out["num_sem"] = parts[1].astype("int32")
        out["week_id"] = out["annee"] * 100 + out["num_sem"]
    else:
        out["week_id"] = out["week_id"].astype("int32")
        if "annee" not in out.columns:
            out["annee"] = (out["week_id"] // 100).astype("int32")
        if "num_sem" not in out.columns:
            out["num_sem"] = (out["week_id"] % 100).astype("int32")
    return out


def prepare_lgb_features(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Iterable[str],
) -> pd.DataFrame:
    """Build a compact, LightGBM-ready feature matrix in the expected order."""
    categorical_set = set(categorical_features)
    X = pd.DataFrame(index=df.index)
    for c in features:
        X[c] = df[c] if c in df.columns else np.nan

    for c in features:
        if c in categorical_set:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1).astype("int32").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X


def build_iterative_state(history_df: pd.DataFrame, y_col: str = "quantite") -> dict:
    """Build the mutable history state consumed by iterative scoring."""
    hist = ensure_time_columns(history_df)
    hist = hist.loc[hist[y_col].notna(), ["week_id", "code_agence", "code_article", "num_sem", y_col]]
    hist = hist.sort_values(["week_id", "code_agence", "code_article"])

    pair_hist: dict[tuple[int, int], list[float]] = {}
    pair_sem_hist: dict[tuple[int, int, int], list[float]] = {}
    agency_hist: dict[int, list[float]] = {}
    article_hist: dict[int, list[float]] = {}
    pair_sum: dict[tuple[int, int], float] = {}

    for _, ag, art, sem, y in hist.itertuples(index=False, name=None):
        ag = int(ag)
        art = int(art)
        sem = int(sem)
        y = float(y)
        pair = (ag, art)
        pair_hist.setdefault(pair, []).append(y)
        pair_sem_hist.setdefault((ag, art, sem), []).append(y)
        agency_hist.setdefault(ag, []).append(y)
        article_hist.setdefault(art, []).append(y)
        pair_sum[pair] = pair_sum.get(pair, 0.0) + y

    return {
        "pair": pair_hist,
        "pair_sem": pair_sem_hist,
        "agency": agency_hist,
        "article": article_hist,
        "pair_sum": pair_sum,
    }


def clone_iterative_state(state: dict) -> dict:
    return {
        "pair": {k: v.copy() for k, v in state["pair"].items()},
        "pair_sem": {k: v.copy() for k, v in state["pair_sem"].items()},
        "agency": {k: v.copy() for k, v in state["agency"].items()},
        "article": {k: v.copy() for k, v in state["article"].items()},
        "pair_sum": state["pair_sum"].copy(),
    }


def _mean(vals: Sequence[float]) -> float:
    return float(np.mean(vals)) if len(vals) else np.nan


def _median(vals: Sequence[float]) -> float:
    return float(np.median(vals)) if len(vals) else np.nan


def _max(vals: Sequence[float]) -> float:
    return float(np.max(vals)) if len(vals) else np.nan


def _std(vals: Sequence[float]) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan


def _zero_rate(vals: Sequence[float]) -> float:
    return float(np.mean(np.asarray(vals, dtype="float64") == 0.0)) if len(vals) else np.nan


def _ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num / den)


def _feature_rows_for_week(
    week_df: pd.DataFrame,
    state: dict,
    features: Sequence[str],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
    rolling_median_windows: Sequence[int],
) -> pd.DataFrame:
    ag_stats = {ag: (_mean(vals), _median(vals)) for ag, vals in state["agency"].items()}
    art_stats = {art: (_mean(vals), _median(vals)) for art, vals in state["article"].items()}
    feature_set = set(features)
    rows = []

    for base in week_df.to_dict("records"):
        ag = int(base["code_agence"])
        art = int(base["code_article"])
        sem = int(base["num_sem"])
        pair = (ag, art)
        vals = state["pair"].get(pair, [])
        sem_vals = state["pair_sem"].get((ag, art, sem), [])

        def lag(n: int) -> float:
            return vals[-n] if len(vals) >= n else np.nan

        def tail(n: int) -> list[float]:
            return vals[-n:] if len(vals) else []

        lag_1 = lag(1)
        lag_2 = lag(2)
        lag_4 = lag(4)
        lag_13 = lag(13)
        lag_52 = lag(52)
        lag_104 = lag(104)
        pair_mean = _mean(vals)
        pair_std = _std(vals)
        recent4 = tail(4)
        prev4 = vals[-8:-4] if len(vals) >= 5 else []
        mean_recent4 = _mean(recent4)
        mean_prev4 = _mean(prev4)
        positive_vals = [v for v in vals if v > 0]
        n_active_weeks = float(len(positive_vals))
        pair_count = float(len(vals))
        pair_active_rate = n_active_weeks / pair_count if pair_count else np.nan
        weeks_since_last_sale = np.nan
        if positive_vals:
            last_positive_idx = max(i for i, v in enumerate(vals) if v > 0)
            weeks_since_last_sale = float(len(vals) - last_positive_idx)

        def active_rate(n: int) -> float:
            tv = tail(n)
            return float(np.mean(np.asarray(tv) > 0)) if len(tv) else np.nan

        def tail_sum(n: int) -> float:
            tv = tail(n)
            return float(np.sum(tv)) if len(tv) else np.nan

        trend_8 = np.nan
        if not pd.isna(mean_prev4):
            trend_8 = min(max((mean_recent4 - mean_prev4) / (mean_prev4 + 1.0), -5.0), 5.0)

        ag_mean, ag_median = ag_stats.get(ag, (np.nan, np.nan))
        art_mean, art_median = art_stats.get(art, (np.nan, np.nan))
        month_num = int(min(12, max(1, math.ceil(sem / 4.333))))
        quarter_num = int(math.ceil(month_num / 3.0))
        dead_from_state = 1 if len(vals) > 0 and state["pair_sum"].get(pair, 0.0) == 0 else 0
        raw_dead = base.get("is_dead_pair", 0)
        base_dead = 0 if pd.isna(raw_dead) else int(raw_dead)

        row = dict(base)
        row.update({
            "is_dead_pair": max(base_dead, dead_from_state),
            "zero_rate_26": _zero_rate(tail(26)),
            "zero_rate_52": _zero_rate(tail(52)),
            "pair_zero_rate_expanding": _zero_rate(vals),
            "active_rate_13": active_rate(13),
            "active_rate_26": active_rate(26),
            "recent_sum_13": tail_sum(13),
            "recent_sum_26": tail_sum(26),
            "lag_1_is_zero": np.nan if pd.isna(lag_1) else float(lag_1 == 0),
            "lag_2_is_zero": np.nan if pd.isna(lag_2) else float(lag_2 == 0),
            "has_lag_1": float(not pd.isna(lag_1)),
            "has_lag_13": float(not pd.isna(lag_13)),
            "has_lag_52": float(not pd.isna(lag_52)),
            "weeks_since_last_sale": weeks_since_last_sale,
            "pair_active_rate_expanding": pair_active_rate,
            "trend_8": trend_8,
            "ratio_n1_vs_mean": _ratio(lag_52, pair_mean),
            "yoy_ratio": _ratio(lag_52, lag_104),
            "pair_mean": pair_mean,
            "pair_median": _median(vals),
            "pair_max": _max(vals),
            "pair_count": pair_count,
            "pair_cv": np.nan if pd.isna(pair_mean) or pair_mean == 0 else pair_std / (pair_mean + 1e-6),
            "sem_mean": _mean(sem_vals),
            "sem_max": _max(sem_vals),
            "sem_median": _median(sem_vals),
            "agence_mean": ag_mean,
            "agence_median": ag_median,
            "article_mean": art_mean,
            "article_median": art_median,
            "n_active_weeks": n_active_weeks,
            "sin_sem": math.sin(2 * math.pi * sem / 52.0),
            "cos_sem": math.cos(2 * math.pi * sem / 52.0),
            "month_num": month_num,
            "quarter_num": quarter_num,
            "weeks_to_year_end": 52 - sem,
            "is_summer_trough": 1 if 30 <= sem <= 35 else 0,
            "is_xmas_trough": 1 if sem >= 50 or sem == 1 else 0,
            "is_q1": 1 if quarter_num == 1 else 0,
            "is_q4": 1 if quarter_num == 4 else 0,
        })

        for n in lags:
            row[f"lag_{n}"] = lag(n)
        for n in rolling_windows:
            tv = tail(n)
            row[f"roll_mean_{n}"] = _mean(tv)
            row[f"roll_std_{n}"] = _std(tv)
        for n in rolling_median_windows:
            row[f"roll_median_{n}"] = _median(tail(n))

        row["roll_mean_4_vs_13"] = _ratio(row.get("roll_mean_4"), row.get("roll_mean_13"))
        row["roll_mean_13_vs_52"] = _ratio(row.get("roll_mean_13"), row.get("roll_mean_52"))
        row["lag1_vs_roll13"] = _ratio(lag_1, row.get("roll_mean_13"))
        row["lag1_minus_roll13"] = (
            np.nan if pd.isna(lag_1) or pd.isna(row.get("roll_mean_13"))
            else lag_1 - row.get("roll_mean_13")
        )
        row["roll_std_13_ratio"] = _ratio(row.get("roll_std_13"), row.get("roll_mean_13"))
        row["sem_mean_vs_pair_mean"] = _ratio(row.get("sem_mean"), pair_mean)

        for c in feature_set:
            row.setdefault(c, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def baseline_quantity_from_features(feat: pd.DataFrame) -> np.ndarray:
    baseline = feat["lag_52"].astype("float64").copy() if "lag_52" in feat.columns else pd.Series(np.nan, index=feat.index)
    for c in [
        "sem_median",
        "sem_mean",
        "roll_mean_13",
        "roll_mean_26",
        "pair_median",
        "pair_mean",
        "article_median",
        "agence_median",
    ]:
        if c in feat.columns:
            baseline = baseline.fillna(feat[c].astype("float64"))
    return baseline.fillna(0.0).clip(lower=0.0).to_numpy()


def append_predictions_to_state(
    state: dict,
    scored_week: pd.DataFrame,
    feed_rounded: bool = True,
) -> None:
    for r in scored_week[["code_agence", "code_article", "num_sem", "prediction", "prediction_int"]].itertuples(index=False, name=None):
        ag, art, sem, prediction, prediction_int = r
        ag = int(ag)
        art = int(art)
        sem = int(sem)
        pair = (ag, art)
        y_feed = float(prediction_int if feed_rounded else prediction)
        state["pair"].setdefault(pair, []).append(y_feed)
        state["pair_sem"].setdefault((ag, art, sem), []).append(y_feed)
        state["agency"].setdefault(ag, []).append(y_feed)
        state["article"].setdefault(art, []).append(y_feed)
        state["pair_sum"][pair] = state["pair_sum"].get(pair, 0.0) + y_feed


def _num_iteration(value):
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def score_iterative_horizon(
    base_state: dict,
    horizon_df: pd.DataFrame,
    zero_model,
    qty_model,
    features: Sequence[str],
    categorical_features: Iterable[str],
    threshold: float,
    label: str,
    zero_iter: int | None = None,
    qty_iter: int | None = None,
    blend_alpha: float = 0.0,
    feed_rounded: bool = True,
    return_features: bool = False,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    rolling_median_windows: Sequence[int] = DEFAULT_ROLLING_MEDIAN_WINDOWS,
) -> pd.DataFrame:
    """Score a horizon recursively, feeding predictions into future features."""
    horizon = ensure_time_columns(horizon_df)
    horizon = horizon.sort_values(["week_id", "code_agence", "code_article"]).reset_index(drop=True)
    if horizon.empty:
        return horizon.copy()

    state = clone_iterative_state(base_state)
    out_parts = []
    zero_iter = _num_iteration(zero_iter)
    qty_iter = _num_iteration(qty_iter)

    for week_id in sorted(horizon["week_id"].unique()):
        week_df = horizon.loc[horizon["week_id"] == week_id].copy()
        feat = _feature_rows_for_week(
            week_df,
            state,
            features,
            lags=lags,
            rolling_windows=rolling_windows,
            rolling_median_windows=rolling_median_windows,
        )
        X = prepare_lgb_features(feat, features, categorical_features)
        p_zero = zero_model.predict(X, num_iteration=zero_iter)
        qty_raw = np.clip(qty_model.predict(X, num_iteration=qty_iter), 0.0, None)
        if blend_alpha > 0:
            baseline = baseline_quantity_from_features(feat)
            qty = (1.0 - blend_alpha) * qty_raw + blend_alpha * baseline
        else:
            qty = qty_raw

        pred = np.where(p_zero > threshold, 0.0, qty)
        if "is_dead_pair" in feat.columns:
            pred = np.where(feat["is_dead_pair"].to_numpy() == 1, 0.0, pred)

        meta_cols = ["semaine", "week_id", "annee", "num_sem", "code_agence", "code_article", "quantite", "is_dead_pair"]
        keep_cols = feat.columns if return_features else [c for c in meta_cols if c in feat.columns]
        scored = feat.loc[:, keep_cols].copy()
        scored["p_zero"] = p_zero
        scored["qty_pred"] = qty
        scored["prediction"] = pred
        scored["prediction_int"] = np.clip(np.round(pred), 0, None).astype("int64")
        scored["split"] = label
        out_parts.append(scored)
        append_predictions_to_state(state, scored, feed_rounded=feed_rounded)

    return pd.concat(out_parts, ignore_index=True)
