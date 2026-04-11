import math

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import PAIR_KEYS, LAGS_ALL, ROLLING_WINDOWS, ROLLING_MEDIAN_WINDOWS, FEATURES


# ============================================================================
# GOLD: feature table + temporal splits
# ============================================================================

@dp.materialized_view(
    name="gold_feature_table",
    comment=(
        "Weekly feature table (train + test) with lags, rolling stats, "
        "expanding pair/agency/article stats, same-week-of-year history, "
        "billing-derived metrics, temporal encodings and categorical dims. "
        "Every window is strictly past (rowsBetween(..., -1))."
    ),
    partition_cols=["annee"],
    table_properties={"quality": "gold"},
)
def gold_feature_table():
    panel = spark.read.table("silver_panel")
    articles_enc = spark.read.table("silver_articles_encoded")
    agences_enc = spark.read.table("silver_agences_encoded")
    fac = spark.read.table("silver_facturation_lagged")

    # Double-cast of target so every window aggregation stays in double-land.
    df = panel.withColumn("y", F.col("quantite").cast("double"))

    pair_order = Window.partitionBy(*PAIR_KEYS).orderBy("week_id")

    # --- 1. Lags -----------------------------------------------------------
    for n in LAGS_ALL:
        df = df.withColumn(f"lag_{n}", F.lag("y", n).over(pair_order))

    # --- 2. Rolling mean / std ---------------------------------------------
    def _lookback(n):
        return (
            Window.partitionBy(*PAIR_KEYS)
            .orderBy("week_id")
            .rowsBetween(-n, -1)
        )

    for n in ROLLING_WINDOWS:
        w = _lookback(n)
        df = (
            df.withColumn(f"roll_mean_{n}", F.avg("y").over(w))
              .withColumn(f"roll_std_{n}", F.stddev("y").over(w))
        )
    for n in ROLLING_MEDIAN_WINDOWS:
        df = df.withColumn(
            f"roll_median_{n}",
            F.expr("percentile_approx(y, 0.5)").over(_lookback(n)),
        )

    # --- 3. Zero rates and short-term trend --------------------------------
    df = df.withColumn(
        "_y_is_zero",
        F.when(F.col("y") == 0, F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    df = (
        df
        .withColumn("zero_rate_26", F.avg("_y_is_zero").over(_lookback(26)))
        .withColumn("zero_rate_52", F.avg("_y_is_zero").over(_lookback(52)))
        .withColumn(
            "pair_zero_rate_expanding",
            F.avg("_y_is_zero").over(
                Window.partitionBy(*PAIR_KEYS)
                .orderBy("week_id")
                .rowsBetween(Window.unboundedPreceding, -1)
            ),
        )
    )

    recent_w = (
        Window.partitionBy(*PAIR_KEYS).orderBy("week_id").rowsBetween(-4, -1)
    )
    prev_w = (
        Window.partitionBy(*PAIR_KEYS).orderBy("week_id").rowsBetween(-8, -5)
    )
    df = (
        df
        .withColumn("_mean_recent4", F.avg("y").over(recent_w))
        .withColumn("_mean_prev4", F.avg("y").over(prev_w))
        .withColumn(
            "trend_8",
            F.when(
                F.col("_mean_prev4").isNull(),
                F.lit(None).cast("double"),
            ).otherwise(
                F.least(
                    F.greatest(
                        (F.col("_mean_recent4") - F.col("_mean_prev4"))
                        / (F.col("_mean_prev4") + F.lit(1.0)),
                        F.lit(-5.0),
                    ),
                    F.lit(5.0),
                )
            ),
        )
        .withColumn(
            "yoy_ratio",
            F.when(
                F.col("lag_104").isNull() | (F.col("lag_104") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("lag_52") / F.col("lag_104")),
        )
    )

    # --- 4. Pair expanding stats -------------------------------------------
    pair_exp = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = (
        df
        .withColumn("pair_mean", F.avg("y").over(pair_exp))
        .withColumn("pair_median", F.expr("percentile_approx(y, 0.5)").over(pair_exp))
        .withColumn("pair_max", F.max("y").over(pair_exp))
        .withColumn("pair_count", F.count("y").over(pair_exp))
        .withColumn("_pair_std", F.stddev("y").over(pair_exp))
        .withColumn(
            "pair_cv",
            F.when(
                F.col("pair_mean").isNull() | (F.col("pair_mean") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("_pair_std") / (F.col("pair_mean") + F.lit(1e-6))),
        )
        .withColumn(
            "ratio_n1_vs_mean",
            F.when(
                F.col("pair_mean").isNull() | (F.col("pair_mean") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("lag_52") / F.col("pair_mean")),
        )
        .withColumn(
            "n_active_weeks",
            F.sum((F.col("y") > 0).cast("double")).over(pair_exp),
        )
    )

    # --- 5. Same-week-of-year stats ----------------------------------------
    season_w = (
        Window.partitionBy(*PAIR_KEYS, "num_sem")
        .orderBy("annee")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = (
        df
        .withColumn("sem_mean", F.avg("y").over(season_w))
        .withColumn("sem_max", F.max("y").over(season_w))
        .withColumn("sem_median", F.expr("percentile_approx(y, 0.5)").over(season_w))
    )

    # --- 6. Agency & article expanding stats -------------------------------
    ag_w = (
        Window.partitionBy("code_agence")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    art_w = (
        Window.partitionBy("code_article")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = (
        df
        .withColumn("agence_mean", F.avg("y").over(ag_w))
        .withColumn("agence_median", F.expr("percentile_approx(y, 0.5)").over(ag_w))
        .withColumn("article_mean", F.avg("y").over(art_w))
        .withColumn("article_median", F.expr("percentile_approx(y, 0.5)").over(art_w))
    )

    # --- 7. Temporal encodings and calendar flags --------------------------
    two_pi = F.lit(2 * math.pi)
    df = (
        df
        .withColumn("sin_sem", F.sin(two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn("cos_sem", F.cos(two_pi * F.col("num_sem") / F.lit(52.0)))
        .withColumn(
            "is_summer_trough",
            ((F.col("num_sem") >= 30) & (F.col("num_sem") <= 35)).cast("tinyint"),
        )
        .withColumn(
            "is_xmas_trough",
            ((F.col("num_sem") >= 50) | (F.col("num_sem") == 1)).cast("tinyint"),
        )
    )

    # --- 8. Join the small silver dims -------------------------------------
    df = (
        df
        .join(articles_enc, PAIR_KEYS, "left")
        .join(agences_enc, "code_agence", "left")
    )

    # --- 9. Join lagged billing on the calendar-month keys ------------------
    df = (
        df
        .withColumn(
            "_join_mois",
            F.least(
                F.lit(12),
                F.greatest(F.lit(1), F.ceil(F.col("num_sem") / F.lit(4.333))),
            ),
        )
        .withColumn("_join_annee", F.col("annee"))
        .join(
            fac,
            ["code_agence", "code_article", "_join_annee", "_join_mois"],
            "left",
        )
        .drop("_join_annee", "_join_mois")
    )

    # --- 10. Final projection ----------------------------------------------
    base_cols = [
        "semaine", "week_id",
        "code_agence", "code_article",
        "quantite", "quantite_raw", "quantite_smooth",
        "is_anomaly", "is_capped", "is_dead_pair",
    ]

    # Fill in any FEATURES missing from `df` as nulls for stable schema.
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
        "quantite is null — consumed by the inference notebook."
    ),
    table_properties={"quality": "gold"},
)
@dp.expect_or_fail("quantite_is_null", "quantite IS NULL")
def gold_test_features():
    return (
        spark.read.table("gold_feature_table")
        .filter((F.col("semaine") >= "2025-27") & (F.col("semaine") <= "2025-52"))
    )
