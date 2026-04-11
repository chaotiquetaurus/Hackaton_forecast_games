import math

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import (
    PAIR_KEYS,
    LAGS_ALL,
    ROLLING_WINDOWS,
    ROLLING_MEDIAN_WINDOWS,
    LAG_BANDS,
    PAIR_SAFE_GAP,
    FEATURES,
)


def _pair_lookback(n):
    return (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(-n, -1)
    )


def _pair_lag_band(start_lag, end_lag):
    return (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(-end_lag, -start_lag)
    )


def _pair_expanding(gap=1):
    return (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -gap)
    )


def _safe_ratio(num, den):
    return F.when(
        den.isNull() | (F.abs(den) < F.lit(1e-9)),
        F.lit(None).cast("double"),
    ).otherwise(num / den)


def _cv(std_col, mean_col):
    return _safe_ratio(std_col, F.abs(mean_col) + F.lit(1e-6))


def _add_window_stats(df, window, prefix):
    mean_col = f"{prefix}_mean"
    std_col = f"{prefix}_std"
    return (
        df
        .withColumn(mean_col, F.avg("y").over(window))
        .withColumn(std_col, F.stddev("y").over(window))
        .withColumn(f"{prefix}_min", F.min("y").over(window))
        .withColumn(f"{prefix}_max", F.max("y").over(window))
        .withColumn(f"{prefix}_sum", F.sum("y").over(window))
        .withColumn(f"{prefix}_zero_rate", F.avg("_y_is_zero").over(window))
        .withColumn(f"{prefix}_active_count", F.sum("_y_is_nonzero").over(window))
        .withColumn(f"{prefix}_nonzero_mean", F.avg("_y_nonzero").over(window))
        .withColumn(f"{prefix}_cv", _cv(F.col(std_col), F.col(mean_col)))
    )


def _add_band_stats(df, window, start_lag, end_lag):
    base = f"band"
    mean_col = f"{base}_mean_{start_lag}_{end_lag}"
    std_col = f"{base}_std_{start_lag}_{end_lag}"
    return (
        df
        .withColumn(mean_col, F.avg("y").over(window))
        .withColumn(std_col, F.stddev("y").over(window))
        .withColumn(f"{base}_min_{start_lag}_{end_lag}", F.min("y").over(window))
        .withColumn(f"{base}_max_{start_lag}_{end_lag}", F.max("y").over(window))
        .withColumn(f"{base}_zero_rate_{start_lag}_{end_lag}", F.avg("_y_is_zero").over(window))
        .withColumn(f"{base}_active_count_{start_lag}_{end_lag}", F.sum("_y_is_nonzero").over(window))
        .withColumn(f"{base}_nonzero_mean_{start_lag}_{end_lag}", F.avg("_y_nonzero").over(window))
        .withColumn(f"{base}_cv_{start_lag}_{end_lag}", _cv(F.col(std_col), F.col(mean_col)))
    )


def _weekly_group_features(df, group_col, out_prefix):
    weekly = (
        df.groupBy(group_col, "week_id")
        .agg(
            F.avg("y").alias("_week_mean"),
            F.avg("_y_is_zero").alias("_week_zero_rate"),
            F.avg("_y_nonzero").alias("_week_nonzero_mean"),
        )
    )
    exp_w = (
        Window.partitionBy(group_col)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    safe_w = (
        Window.partitionBy(group_col)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -PAIR_SAFE_GAP)
    )
    return (
        weekly
        .withColumn(f"{out_prefix}_mean", F.avg("_week_mean").over(exp_w))
        .withColumn(f"{out_prefix}_median", F.expr("percentile_approx(_week_mean, 0.5)").over(exp_w))
        .withColumn(f"{out_prefix}_mean_lag26", F.avg("_week_mean").over(safe_w))
        .withColumn(f"{out_prefix}_zero_rate_lag26", F.avg("_week_zero_rate").over(safe_w))
        .withColumn(f"{out_prefix}_nonzero_mean_lag26", F.avg("_week_nonzero_mean").over(safe_w))
        .select(
            group_col,
            "week_id",
            f"{out_prefix}_mean",
            f"{out_prefix}_median",
            f"{out_prefix}_mean_lag26",
            f"{out_prefix}_zero_rate_lag26",
            f"{out_prefix}_nonzero_mean_lag26",
        )
    )


# ============================================================================
# GOLD: feature table + temporal splits
# ============================================================================

@dp.materialized_view(
    name="gold_feature_table",
    comment=(
        "Wide weekly feature table (train + test) with 250 candidate features: "
        "lags, rolling stats, horizon-safe lag bands, expanding pair stats, "
        "seasonal history, weekly agency/article context, billing metrics, "
        "calendar flags and categorical dimensions. Target-derived windows are "
        "strictly past-only."
    ),
    partition_cols=["annee"],
    table_properties={"quality": "gold"},
)
def gold_feature_table():
    panel = spark.read.table("silver_panel")
    articles_enc = spark.read.table("silver_articles_encoded")
    agences_enc = spark.read.table("silver_agences_encoded")
    fac = spark.read.table("silver_facturation_lagged")

    df = (
        panel
        .withColumn("y", F.col("quantite").cast("double"))
        .withColumn(
            "_y_is_zero",
            F.when(F.col("quantite").isNull(), F.lit(None).cast("double"))
             .when(F.col("quantite") == 0, F.lit(1.0))
             .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "_y_is_nonzero",
            F.when(F.col("quantite").isNull(), F.lit(None).cast("double"))
             .when(F.col("quantite") > 0, F.lit(1.0))
             .otherwise(F.lit(0.0)),
        )
        .withColumn(
            "_y_nonzero",
            F.when(F.col("quantite") > 0, F.col("quantite").cast("double"))
             .otherwise(F.lit(None).cast("double")),
        )
    )

    pair_order = Window.partitionBy(*PAIR_KEYS).orderBy("week_id")
    df = df.withColumn("_pair_row_idx", F.row_number().over(pair_order))

    # 1. Direct lags.
    for n in LAGS_ALL:
        df = df.withColumn(f"lag_{n}", F.lag("y", n).over(pair_order))

    # 2. Classical rolling features, useful for screening and diagnostics.
    for n in ROLLING_WINDOWS:
        df = _add_window_stats(df, _pair_lookback(n), f"roll")
        rename_pairs = {
            "roll_mean": f"roll_mean_{n}",
            "roll_std": f"roll_std_{n}",
            "roll_min": f"roll_min_{n}",
            "roll_max": f"roll_max_{n}",
            "roll_sum": f"roll_sum_{n}",
            "roll_zero_rate": f"roll_zero_rate_{n}",
            "roll_active_count": f"roll_active_count_{n}",
            "roll_nonzero_mean": f"roll_nonzero_mean_{n}",
            "roll_cv": f"roll_cv_{n}",
        }
        for old, new in rename_pairs.items():
            df = df.withColumnRenamed(old, new)

    for n in ROLLING_MEDIAN_WINDOWS:
        df = df.withColumn(
            f"roll_median_{n}",
            F.expr("percentile_approx(y, 0.5)").over(_pair_lookback(n)),
        )

    # 3. Horizon-safe lag bands. These are eligible for the final elite model.
    for start_lag, end_lag in LAG_BANDS:
        df = _add_band_stats(df, _pair_lag_band(start_lag, end_lag), start_lag, end_lag)

    # 4. Pair-level expanding stats and lag-26-safe expanding stats.
    pair_exp = _pair_expanding(1)
    pair_safe = _pair_expanding(PAIR_SAFE_GAP)
    df = (
        df
        .withColumn("pair_mean", F.avg("y").over(pair_exp))
        .withColumn("pair_median", F.expr("percentile_approx(y, 0.5)").over(pair_exp))
        .withColumn("pair_max", F.max("y").over(pair_exp))
        .withColumn("pair_min", F.min("y").over(pair_exp))
        .withColumn("pair_count", F.count("y").over(pair_exp))
        .withColumn("pair_std", F.stddev("y").over(pair_exp))
        .withColumn("pair_cv", _cv(F.col("pair_std"), F.col("pair_mean")))
        .withColumn("pair_zero_rate_expanding", F.avg("_y_is_zero").over(pair_exp))
        .withColumn("n_active_weeks", F.sum("_y_is_nonzero").over(pair_exp))
        .withColumn("pair_nonzero_mean", F.avg("_y_nonzero").over(pair_exp))
        .withColumn("pair_nonzero_share", _safe_ratio(F.col("n_active_weeks"), F.col("pair_count")))
        .withColumn(
            "_last_nonzero_row_idx",
            F.max(F.when(F.col("y") > 0, F.col("_pair_row_idx"))).over(pair_exp),
        )
        .withColumn("pair_last_nonzero_gap", F.col("_pair_row_idx") - F.col("_last_nonzero_row_idx"))
        .withColumn("pair_mean_lag26", F.avg("y").over(pair_safe))
        .withColumn("pair_median_lag26", F.expr("percentile_approx(y, 0.5)").over(pair_safe))
        .withColumn("pair_max_lag26", F.max("y").over(pair_safe))
        .withColumn("pair_min_lag26", F.min("y").over(pair_safe))
        .withColumn("pair_std_lag26", F.stddev("y").over(pair_safe))
        .withColumn("pair_cv_lag26", _cv(F.col("pair_std_lag26"), F.col("pair_mean_lag26")))
        .withColumn("pair_zero_rate_lag26", F.avg("_y_is_zero").over(pair_safe))
        .withColumn("pair_active_weeks_lag26", F.sum("_y_is_nonzero").over(pair_safe))
        .withColumn("pair_count_lag26", F.count("y").over(pair_safe))
        .withColumn("pair_nonzero_mean_lag26", F.avg("_y_nonzero").over(pair_safe))
        .withColumn(
            "pair_nonzero_share_lag26",
            _safe_ratio(F.col("pair_active_weeks_lag26"), F.col("pair_count_lag26")),
        )
        .withColumn("yoy_ratio", _safe_ratio(F.col("lag_52"), F.col("lag_104")))
        .withColumn(
            "ratio_lag52_vs_pair_mean_lag26",
            _safe_ratio(F.col("lag_52"), F.col("pair_mean_lag26")),
        )
        .withColumn(
            "ratio_band26_52_vs_pair_mean_lag26",
            _safe_ratio(F.col("band_mean_26_52"), F.col("pair_mean_lag26")),
        )
        .withColumn(
            "trend_band_26_52_vs_52_104",
            _safe_ratio(
                F.col("band_mean_26_52") - F.col("band_mean_52_104"),
                F.abs(F.col("band_mean_52_104")) + F.lit(1.0),
            ),
        )
    )

    # 5. Same-week-of-year history.
    season_order = Window.partitionBy(*PAIR_KEYS, "num_sem").orderBy("annee")
    season_w = season_order.rowsBetween(Window.unboundedPreceding, -1)
    df = (
        df
        .withColumn("sem_mean", F.avg("y").over(season_w))
        .withColumn("sem_max", F.max("y").over(season_w))
        .withColumn("sem_min", F.min("y").over(season_w))
        .withColumn("sem_median", F.expr("percentile_approx(y, 0.5)").over(season_w))
        .withColumn("sem_std", F.stddev("y").over(season_w))
        .withColumn("sem_count", F.count("y").over(season_w))
        .withColumn("sem_zero_rate", F.avg("_y_is_zero").over(season_w))
        .withColumn("sem_nonzero_mean", F.avg("_y_nonzero").over(season_w))
        .withColumn("sem_lag_y1", F.lag("y", 1).over(season_order))
        .withColumn("sem_lag_y2", F.lag("y", 2).over(season_order))
        .withColumn("sem_yoy_ratio_1", _safe_ratio(F.col("sem_lag_y1"), F.col("sem_lag_y2")))
        .withColumn("sem_yoy_ratio_2", _safe_ratio(F.col("sem_lag_y2"), F.lag("y", 3).over(season_order)))
        .withColumn(
            "sem_trend_yoy",
            _safe_ratio(F.col("sem_lag_y1") - F.col("sem_lag_y2"), F.abs(F.col("sem_lag_y2")) + F.lit(1.0)),
        )
    )

    # 6. Weekly agency and article context, computed on weekly aggregates so a
    # window never steps through arbitrary rows from the same week.
    agency_features = _weekly_group_features(df, "code_agence", "agence")
    article_features = _weekly_group_features(df, "code_article", "article")
    df = (
        df
        .join(agency_features, ["code_agence", "week_id"], "left")
        .join(article_features, ["code_article", "week_id"], "left")
    )

    # 7. Calendar encodings and event flags.
    two_pi = F.lit(2 * math.pi)
    month_approx = F.least(
        F.lit(12),
        F.greatest(F.lit(1), F.ceil(F.col("num_sem") / F.lit(4.333)).cast("int")),
    )
    df = (
        df
        .withColumn("mois_approx", month_approx.cast("int"))
        .withColumn("trimestre", F.least(F.lit(4), F.ceil(F.col("num_sem") / F.lit(13.0)).cast("int")))
        .withColumn("week_of_quarter", ((F.col("num_sem") - F.lit(1)) % F.lit(13)) + F.lit(1))
        .withColumn("sin_sem", F.sin(two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("cos_sem", F.cos(two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("sin_sem_2", F.sin(F.lit(2.0) * two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("cos_sem_2", F.cos(F.lit(2.0) * two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("sin_sem_3", F.sin(F.lit(3.0) * two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("cos_sem_3", F.cos(F.lit(3.0) * two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("sin_sem_4", F.sin(F.lit(4.0) * two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("cos_sem_4", F.cos(F.lit(4.0) * two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("is_summer_trough", ((F.col("num_sem") >= 30) & (F.col("num_sem") <= 35)).cast("tinyint"))
        .withColumn("is_xmas_trough", ((F.col("num_sem") >= 50) | (F.col("num_sem") == 1)).cast("tinyint"))
        .withColumn("is_august", (F.col("mois_approx") == 8).cast("tinyint"))
        .withColumn("is_december", (F.col("mois_approx") == 12).cast("tinyint"))
        .withColumn("is_quarter_end", F.col("num_sem").isin(13, 26, 39, 52).cast("tinyint"))
        .withColumn(
            "is_peak_spring",
            (((F.col("num_sem") >= 11) & (F.col("num_sem") <= 13)) |
             ((F.col("num_sem") >= 24) & (F.col("num_sem") <= 25))).cast("tinyint"),
        )
        .withColumn(
            "is_peak_autumn",
            (((F.col("num_sem") >= 36) & (F.col("num_sem") <= 40)) |
             (F.col("num_sem") == 46)).cast("tinyint"),
        )
        .withColumn(
            "weeks_to_summer",
            F.least(F.abs(F.col("num_sem") - F.lit(32)), F.abs(F.col("num_sem") - F.lit(33))).cast("double"),
        )
        .withColumn(
            "weeks_to_xmas",
            F.least(
                F.abs(F.col("num_sem") - F.lit(51)),
                F.abs(F.col("num_sem") - F.lit(52)),
                F.abs(F.col("num_sem") - F.lit(1)),
            ).cast("double"),
        )
    )

    # 8. Join static dimensions and lagged billing.
    df = (
        df
        .join(articles_enc, PAIR_KEYS, "left")
        .join(agences_enc, "code_agence", "left")
        .withColumn("_join_mois", F.col("mois_approx").cast("int"))
        .withColumn("_join_annee", F.col("annee").cast("int"))
        .join(
            fac,
            ["code_agence", "code_article", "_join_annee", "_join_mois"],
            "left",
        )
        .drop("_join_annee", "_join_mois")
    )

    base_cols = [
        "semaine", "week_id",
        "code_agence", "code_article",
        "quantite", "quantite_raw", "quantite_smooth",
        "is_anomaly", "is_capped", "is_dead_pair",
    ]

    for c in FEATURES:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))

    return df.select(*base_cols, *FEATURES)


# ---------------------------------------------------------------------------
# Gold splits
# ---------------------------------------------------------------------------

@dp.materialized_view(
    name="gold_train",
    comment="Training split: weeks strictly before 2025-W01.",
    table_properties={"quality": "gold"},
)
@dp.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_train():
    return spark.read.table("gold_feature_table").filter(F.col("semaine") < "2025-01")


@dp.materialized_view(
    name="gold_validation",
    comment="Validation split: 2025-W01 .. 2025-W26.",
    table_properties={"quality": "gold"},
)
@dp.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_validation():
    return (
        spark.read.table("gold_feature_table")
        .filter((F.col("semaine") >= "2025-01") & (F.col("semaine") <= "2025-26"))
    )


@dp.materialized_view(
    name="gold_test_features",
    comment=(
        "Feature rows for the test period (2025-W27 .. 2025-W52). "
        "quantite is null; consumed by the inference notebook."
    ),
    table_properties={"quality": "gold"},
)
@dp.expect_or_fail("quantite_is_null", "quantite IS NULL")
def gold_test_features():
    return (
        spark.read.table("gold_feature_table")
        .filter((F.col("semaine") >= "2025-27") & (F.col("semaine") <= "2025-52"))
    )
