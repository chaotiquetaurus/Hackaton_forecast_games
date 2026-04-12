import math

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import (
    PAIR_KEYS,
    LAGS_ALL,
    ROLLING_WINDOWS,
    ROLLING_MEDIAN_WINDOWS,
    FEATURES,
    TRAIN_END_WEEK_ID,
    VAL_START_WEEK_ID,
    VAL_END_WEEK_ID,
    INTERNAL_TEST_START_WEEK_ID,
    INTERNAL_TEST_END_WEEK_ID,
    FINAL_INFERENCE_START_WEEK_ID,
    FINAL_INFERENCE_END_WEEK_ID,
)


# ============================================================================
# GOLD: feature table + temporal splits
# ============================================================================

def _build_feature_frame(panel, articles_enc, agences_enc, fac, history_end_week_id, split_name):
    """Build features using only targets up to `history_end_week_id`.

    This lets validation/test/final-inference rows mimic a 26-week block
    forecast: target values inside the scored horizon are masked before lags,
    rolling stats, expanding stats and zero rates are computed.
    """
    df = panel.withColumn(
        "y",
        F.when(
            F.col("week_id") <= F.lit(history_end_week_id),
            F.col("quantite").cast("double"),
        ).otherwise(F.lit(None).cast("double")),
    ).withColumn("split_name", F.lit(split_name))

    pair_order = Window.partitionBy(*PAIR_KEYS).orderBy("week_id")

    # --- 1. Lags -----------------------------------------------------------
    for n in LAGS_ALL:
        df = df.withColumn(f"lag_{n}", F.lag("y", n).over(pair_order))

    # --- 2. Rolling mean / std --------------------------------------------
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
        F.when(F.col("y").isNull(), F.lit(None).cast("double"))
         .when(F.col("y") == 0, F.lit(1.0))
         .otherwise(F.lit(0.0)),
    ).withColumn(
        "_y_is_positive",
        F.when(F.col("y").isNull(), F.lit(None).cast("double"))
         .when(F.col("y") > 0, F.lit(1.0))
         .otherwise(F.lit(0.0)),
    )
    df = (
        df
        .withColumn("zero_rate_26", F.avg("_y_is_zero").over(_lookback(26)))
        .withColumn("zero_rate_52", F.avg("_y_is_zero").over(_lookback(52)))
        .withColumn("active_rate_13", F.avg("_y_is_positive").over(_lookback(13)))
        .withColumn("active_rate_26", F.avg("_y_is_positive").over(_lookback(26)))
        .withColumn("recent_sum_13", F.sum("y").over(_lookback(13)))
        .withColumn("recent_sum_26", F.sum("y").over(_lookback(26)))
        .withColumn(
            "pair_zero_rate_expanding",
            F.avg("_y_is_zero").over(
                Window.partitionBy(*PAIR_KEYS)
                .orderBy("week_id")
                .rowsBetween(Window.unboundedPreceding, -1)
            ),
        )
        .withColumn("lag_1_is_zero", F.when(F.col("lag_1").isNull(), None).otherwise((F.col("lag_1") == 0).cast("double")))
        .withColumn("lag_2_is_zero", F.when(F.col("lag_2").isNull(), None).otherwise((F.col("lag_2") == 0).cast("double")))
        .withColumn("has_lag_1", F.col("lag_1").isNotNull().cast("double"))
        .withColumn("has_lag_13", F.col("lag_13").isNotNull().cast("double"))
        .withColumn("has_lag_52", F.col("lag_52").isNotNull().cast("double"))
    )

    recent_w = Window.partitionBy(*PAIR_KEYS).orderBy("week_id").rowsBetween(-4, -1)
    prev_w = Window.partitionBy(*PAIR_KEYS).orderBy("week_id").rowsBetween(-8, -5)
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
    df = df.withColumn("_pair_obs_idx", F.row_number().over(pair_order) - F.lit(1))
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
            "n_active_weeks",
            F.sum((F.col("y") > 0).cast("double")).over(pair_exp),
        )
        .withColumn("_last_positive_obs_idx", F.max(F.when(F.col("y") > 0, F.col("_pair_obs_idx"))).over(pair_exp))
        .withColumn(
            "weeks_since_last_sale",
            F.when(F.col("_last_positive_obs_idx").isNull(), None)
             .otherwise(F.col("_pair_obs_idx") - F.col("_last_positive_obs_idx")),
        )
        .withColumn(
            "pair_active_rate_expanding",
            F.when(
                F.col("pair_count").isNull() | (F.col("pair_count") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("n_active_weeks") / F.col("pair_count")),
        )
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
        .withColumn(
            "roll_mean_4_vs_13",
            F.when(
                F.col("roll_mean_13").isNull() | (F.col("roll_mean_13") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("roll_mean_4") / F.col("roll_mean_13")),
        )
        .withColumn(
            "roll_mean_13_vs_52",
            F.when(
                F.col("roll_mean_52").isNull() | (F.col("roll_mean_52") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("roll_mean_13") / F.col("roll_mean_52")),
        )
        .withColumn(
            "lag1_vs_roll13",
            F.when(
                F.col("roll_mean_13").isNull() | (F.col("roll_mean_13") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("lag_1") / F.col("roll_mean_13")),
        )
        .withColumn(
            "lag1_minus_roll13",
            F.when(
                F.col("lag_1").isNull() | F.col("roll_mean_13").isNull(),
                F.lit(None).cast("double"),
            ).otherwise(F.col("lag_1") - F.col("roll_mean_13")),
        )
        .withColumn(
            "roll_std_13_ratio",
            F.when(
                F.col("roll_mean_13").isNull() | (F.col("roll_mean_13") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("roll_std_13") / F.col("roll_mean_13")),
        )
        .withColumn(
            "sem_mean_vs_pair_mean",
            F.when(
                F.col("pair_mean").isNull() | (F.col("pair_mean") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("sem_mean") / F.col("pair_mean")),
        )
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
            "month_num",
            F.least(F.lit(12), F.greatest(F.lit(1), F.ceil(F.col("num_sem") / F.lit(4.333)))),
        )
        .withColumn("quarter_num", F.ceil(F.col("month_num") / F.lit(3.0)))
        .withColumn("weeks_to_year_end", F.lit(52) - F.col("num_sem"))
        .withColumn(
            "is_summer_trough",
            ((F.col("num_sem") >= 30) & (F.col("num_sem") <= 35)).cast("tinyint"),
        )
        .withColumn(
            "is_xmas_trough",
            ((F.col("num_sem") >= 50) | (F.col("num_sem") == 1)).cast("tinyint"),
        )
        .withColumn("is_q1", (F.col("quarter_num") == 1).cast("tinyint"))
        .withColumn("is_q4", (F.col("quarter_num") == 4).cast("tinyint"))
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
        "semaine", "week_id", "split_name",
        "code_agence", "code_article",
        "quantite", "quantite_raw", "quantite_smooth",
        "is_anomaly", "is_capped", "is_dead_pair",
    ]

    # Fill in any FEATURES missing from `df` as nulls for stable schema.
    for c in FEATURES:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))

    return df.select(*base_cols, *FEATURES)


@dp.materialized_view(
    name="gold_feature_table",
    comment=(
        "Weekly feature table with leakage-safe train, validation, internal "
        "test and final-inference splits. Validation/test/final horizons mask "
        "their own target values before lag and rolling features are computed."
    ),
    partition_cols=["annee"],
    table_properties={"quality": "gold"},
)
def gold_feature_table():
    panel = spark.read.table("silver_panel")
    articles_enc = spark.read.table("silver_articles_encoded")
    agences_enc = spark.read.table("silver_agences_encoded")
    fac = spark.read.table("silver_facturation_lagged")

    train = (
        _build_feature_frame(panel, articles_enc, agences_enc, fac, TRAIN_END_WEEK_ID, "train")
        .filter(F.col("week_id") <= F.lit(TRAIN_END_WEEK_ID))
    )
    validation = (
        _build_feature_frame(panel, articles_enc, agences_enc, fac, TRAIN_END_WEEK_ID, "validation")
        .filter(
            (F.col("week_id") >= F.lit(VAL_START_WEEK_ID))
            & (F.col("week_id") <= F.lit(VAL_END_WEEK_ID))
        )
    )
    internal_test = (
        _build_feature_frame(panel, articles_enc, agences_enc, fac, VAL_END_WEEK_ID, "internal_test")
        .filter(
            (F.col("week_id") >= F.lit(INTERNAL_TEST_START_WEEK_ID))
            & (F.col("week_id") <= F.lit(INTERNAL_TEST_END_WEEK_ID))
        )
    )
    final_inference = (
        _build_feature_frame(
            panel,
            articles_enc,
            agences_enc,
            fac,
            INTERNAL_TEST_END_WEEK_ID,
            "final_inference",
        )
        .filter(
            (F.col("week_id") >= F.lit(FINAL_INFERENCE_START_WEEK_ID))
            & (F.col("week_id") <= F.lit(FINAL_INFERENCE_END_WEEK_ID))
        )
    )

    return (
        train
        .unionByName(validation)
        .unionByName(internal_test)
        .unionByName(final_inference)
    )


# ---------------------------------------------------------------------------
# Gold splits
# ---------------------------------------------------------------------------

@dp.materialized_view(
    name="gold_train",
    comment="Training split: weeks up to 2024-W26.",
    table_properties={"quality": "gold"},
)
@dp.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_train():
    return spark.read.table("gold_feature_table").filter(F.col("split_name") == "train")


@dp.materialized_view(
    name="gold_validation",
    comment="Validation split: 2024-W27 .. 2024-W52, with masked-horizon features.",
    table_properties={"quality": "gold"},
)
@dp.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_validation():
    return spark.read.table("gold_feature_table").filter(F.col("split_name") == "validation")


@dp.materialized_view(
    name="gold_internal_test",
    comment="Labelled internal test split: 2025-W01 .. 2025-W26, never used for training.",
    table_properties={"quality": "gold"},
)
@dp.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_internal_test():
    return spark.read.table("gold_feature_table").filter(F.col("split_name") == "internal_test")


@dp.materialized_view(
    name="gold_test_features",
    comment=(
        "Final-inference feature rows for the leaderboard period "
        "(2025-W27 .. 2025-W52). quantite is null."
    ),
    table_properties={"quality": "gold"},
)
@dp.expect_or_fail("quantite_is_null", "quantite IS NULL")
def gold_test_features():
    return spark.read.table("gold_feature_table").filter(F.col("split_name") == "final_inference")
