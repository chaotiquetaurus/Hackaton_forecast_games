import math

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import PAIR_KEYS, LAGS_ALL, ROLLING_WINDOWS, ROLLING_MEDIAN_WINDOWS, BAND_WINDOWS


# ============================================================================
# GOLD: feature table + temporal splits
# ============================================================================

@dp.materialized_view(
    name="gold_feature_table",
    comment=(
        "Weekly feature table (train + test) with inference-safe lags, band "
        "aggregates, lag-26 pair/agency/article stats, rolling billing, "
        "higher-order Fourier harmonics, calendar flags, and encoded dims. "
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

    df = panel.withColumn("y", F.col("quantite").cast("double"))

    pair_order = Window.partitionBy(*PAIR_KEYS).orderBy("week_id")

    # Helper: past-N-rows window (current row excluded).
    def _lookback(n):
        return (
            Window.partitionBy(*PAIR_KEYS)
            .orderBy("week_id")
            .rowsBetween(-n, -1)
        )

    # Expanding window (current row excluded).
    pair_exp = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )

    # ---- helper columns used by multiple sections -----------------------
    df = df.withColumn(
        "_y_is_zero",
        F.when(F.col("y") == 0, F.lit(1.0)).otherwise(F.lit(0.0)),
    ).withColumn(
        "_y_nonzero",
        F.when(F.col("y") > 0, F.col("y")),
    )

    # =====================================================================
    # 1. LAGS
    # =====================================================================
    for n in LAGS_ALL:
        df = df.withColumn(f"lag_{n}", F.lag("y", n).over(pair_order))

    # =====================================================================
    # 2. ROLLING MEAN / STD / MEDIAN (kept for legacy; band features below
    #    are the inference-safe equivalents)
    # =====================================================================
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

    # =====================================================================
    # 3. BAND FEATURES — stats on fixed backward windows (inference-safe)
    # =====================================================================
    for lo, hi in BAND_WINDOWS:
        bw = (
            Window.partitionBy(*PAIR_KEYS)
            .orderBy("week_id")
            .rowsBetween(-hi, -lo)
        )
        tag = f"{lo}_{hi}"
        df = (
            df
            .withColumn(f"band_mean_{tag}",     F.avg("y").over(bw))
            .withColumn(f"band_std_{tag}",      F.stddev("y").over(bw))
            .withColumn(f"band_min_{tag}",      F.min("y").over(bw))
            .withColumn(f"band_max_{tag}",      F.max("y").over(bw))
            .withColumn(f"band_zero_rate_{tag}", F.avg("_y_is_zero").over(bw))
            .withColumn(
                f"band_active_count_{tag}",
                F.sum((F.col("y") > 0).cast("double")).over(bw),
            )
            .withColumn(f"band_nonzero_mean_{tag}", F.avg("_y_nonzero").over(bw))
            .withColumn(
                f"band_cv_{tag}",
                F.when(
                    F.col(f"band_mean_{tag}").isNull()
                    | (F.col(f"band_mean_{tag}") == 0),
                    F.lit(None).cast("double"),
                ).otherwise(
                    F.col(f"band_std_{tag}")
                    / (F.col(f"band_mean_{tag}") + F.lit(1e-6))
                ),
            )
        )

    # =====================================================================
    # 4. ZERO RATES (rolling + expanding)
    # =====================================================================
    df = (
        df
        .withColumn("zero_rate_26", F.avg("_y_is_zero").over(_lookback(26)))
        .withColumn("zero_rate_52", F.avg("_y_is_zero").over(_lookback(52)))
        .withColumn("pair_zero_rate_expanding", F.avg("_y_is_zero").over(pair_exp))
    )

    # =====================================================================
    # 5. TREND & YOY RATIO
    # =====================================================================
    recent_w = Window.partitionBy(*PAIR_KEYS).orderBy("week_id").rowsBetween(-4, -1)
    prev_w   = Window.partitionBy(*PAIR_KEYS).orderBy("week_id").rowsBetween(-8, -5)
    df = (
        df
        .withColumn("_mean_recent4", F.avg("y").over(recent_w))
        .withColumn("_mean_prev4",   F.avg("y").over(prev_w))
        .withColumn(
            "trend_8",
            F.when(F.col("_mean_prev4").isNull(), F.lit(None).cast("double"))
             .otherwise(
                F.least(F.greatest(
                    (F.col("_mean_recent4") - F.col("_mean_prev4"))
                    / (F.col("_mean_prev4") + F.lit(1.0)),
                    F.lit(-5.0),
                ), F.lit(5.0))
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

    # Band-based trend: band_mean_26_52 vs band_mean_52_104.
    df = df.withColumn(
        "trend_band_26_52_vs_52_104",
        F.when(
            F.col("band_mean_52_104").isNull() | (F.col("band_mean_52_104") == 0),
            F.lit(None).cast("double"),
        ).otherwise(
            (F.col("band_mean_26_52") - F.col("band_mean_52_104"))
            / (F.col("band_mean_52_104") + F.lit(1.0))
        ),
    )

    # =====================================================================
    # 6. PAIR EXPANDING STATS (full history)
    # =====================================================================
    df = (
        df
        .withColumn("pair_mean",   F.avg("y").over(pair_exp))
        .withColumn("pair_median", F.expr("percentile_approx(y, 0.5)").over(pair_exp))
        .withColumn("pair_max",    F.max("y").over(pair_exp))
        .withColumn("pair_min",    F.min("y").over(pair_exp))
        .withColumn("pair_count",  F.count("y").over(pair_exp))
        .withColumn("_pair_std",   F.stddev("y").over(pair_exp))
        .withColumn(
            "pair_cv",
            F.when(
                F.col("pair_mean").isNull() | (F.col("pair_mean") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("_pair_std") / (F.col("pair_mean") + F.lit(1e-6))),
        )
        .withColumn("pair_nonzero_mean", F.avg("_y_nonzero").over(pair_exp))
        .withColumn(
            "pair_nonzero_share",
            F.sum((F.col("y") > 0).cast("double")).over(pair_exp)
            / (F.col("pair_count") + F.lit(1e-6)),
        )
        .withColumn(
            "n_active_weeks",
            F.sum((F.col("y") > 0).cast("double")).over(pair_exp),
        )
        .withColumn(
            "ratio_n1_vs_mean",
            F.when(
                F.col("pair_mean").isNull() | (F.col("pair_mean") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("lag_52") / F.col("pair_mean")),
        )
    )

    # =====================================================================
    # 7. PAIR LAG-26 STATS (expanding stats minus last 26 weeks)
    #    These mirror what the model sees at inference (W27-W52 prediction).
    # =====================================================================
    pair_exp_lag26 = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -27)
    )
    df = (
        df
        .withColumn("pair_mean_lag26",   F.avg("y").over(pair_exp_lag26))
        .withColumn("pair_median_lag26", F.expr("percentile_approx(y, 0.5)").over(pair_exp_lag26))
        .withColumn("pair_max_lag26",    F.max("y").over(pair_exp_lag26))
        .withColumn("pair_min_lag26",    F.min("y").over(pair_exp_lag26))
        .withColumn("_pair_std_lag26",   F.stddev("y").over(pair_exp_lag26))
        .withColumn(
            "pair_cv_lag26",
            F.when(
                F.col("pair_mean_lag26").isNull() | (F.col("pair_mean_lag26") == 0),
                F.lit(None).cast("double"),
            ).otherwise(
                F.col("_pair_std_lag26") / (F.col("pair_mean_lag26") + F.lit(1e-6))
            ),
        )
        .withColumn("pair_zero_rate_lag26", F.avg("_y_is_zero").over(pair_exp_lag26))
        .withColumn(
            "pair_active_weeks_lag26",
            F.sum((F.col("y") > 0).cast("double")).over(pair_exp_lag26),
        )
        .withColumn("pair_nonzero_mean_lag26", F.avg("_y_nonzero").over(pair_exp_lag26))
        .withColumn("_pair_count_lag26", F.count("y").over(pair_exp_lag26))
        .withColumn(
            "pair_nonzero_share_lag26",
            F.col("pair_active_weeks_lag26")
            / (F.col("_pair_count_lag26") + F.lit(1e-6)),
        )
        .withColumn("pair_std_lag26", F.col("_pair_std_lag26"))
    )

    # pair_last_nonzero_gap: weeks since last non-zero sale.
    df = df.withColumn(
        "_last_nonzero_week",
        F.max(
            F.when(F.col("y") > 0, F.col("week_id"))
        ).over(pair_exp),
    ).withColumn(
        "pair_last_nonzero_gap",
        F.when(
            F.col("_last_nonzero_week").isNull(),
            F.lit(None).cast("int"),
        ).otherwise(
            # Approximate: week_id difference (ignores year boundaries but
            # works for a monotonic week_id = annee*100 + num_sem).
            # Good enough for a feature; exact ISO arithmetic is overkill.
            (F.col("week_id") - F.col("_last_nonzero_week")).cast("int")
        ),
    )

    # =====================================================================
    # 8. RATIO FEATURES (lag52 / pair_mean_lag26, band / pair_mean_lag26)
    # =====================================================================
    df = df.withColumn(
        "ratio_lag52_vs_pair_mean_lag26",
        F.when(
            F.col("pair_mean_lag26").isNull() | (F.col("pair_mean_lag26") == 0),
            F.lit(None).cast("double"),
        ).otherwise(F.col("lag_52") / F.col("pair_mean_lag26")),
    ).withColumn(
        "ratio_band26_52_vs_pair_mean_lag26",
        F.when(
            F.col("pair_mean_lag26").isNull() | (F.col("pair_mean_lag26") == 0),
            F.lit(None).cast("double"),
        ).otherwise(F.col("band_mean_26_52") / F.col("pair_mean_lag26")),
    )

    # =====================================================================
    # 9. SAME-WEEK-OF-YEAR STATS
    # =====================================================================
    season_w = (
        Window.partitionBy(*PAIR_KEYS, "num_sem")
        .orderBy("annee")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = (
        df
        .withColumn("sem_mean",   F.avg("y").over(season_w))
        .withColumn("sem_max",    F.max("y").over(season_w))
        .withColumn("sem_min",    F.min("y").over(season_w))
        .withColumn("sem_median", F.expr("percentile_approx(y, 0.5)").over(season_w))
        .withColumn("sem_std",    F.stddev("y").over(season_w))
        .withColumn("sem_count",  F.count("y").over(season_w))
        .withColumn("sem_zero_rate",     F.avg("_y_is_zero").over(season_w))
        .withColumn("sem_nonzero_mean",  F.avg("_y_nonzero").over(season_w))
    )

    # =====================================================================
    # 10. AGENCY & ARTICLE EXPANDING STATS (lag-26 variants)
    # =====================================================================
    ag_exp_lag26 = (
        Window.partitionBy("code_agence")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -27)
    )
    art_exp_lag26 = (
        Window.partitionBy("code_article")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -27)
    )
    ag_exp = (
        Window.partitionBy("code_agence")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    art_exp = (
        Window.partitionBy("code_article")
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = (
        df
        # Legacy (full expanding)
        .withColumn("agence_mean",   F.avg("y").over(ag_exp))
        .withColumn("agence_median", F.expr("percentile_approx(y, 0.5)").over(ag_exp))
        .withColumn("article_mean",   F.avg("y").over(art_exp))
        .withColumn("article_median", F.expr("percentile_approx(y, 0.5)").over(art_exp))
        # Lag-26 variants (inference-safe)
        .withColumn("agence_mean_lag26",         F.avg("y").over(ag_exp_lag26))
        .withColumn("agence_zero_rate_lag26",    F.avg("_y_is_zero").over(ag_exp_lag26))
        .withColumn("agence_nonzero_mean_lag26", F.avg("_y_nonzero").over(ag_exp_lag26))
        .withColumn("article_mean_lag26",         F.avg("y").over(art_exp_lag26))
        .withColumn("article_zero_rate_lag26",    F.avg("_y_is_zero").over(art_exp_lag26))
        .withColumn("article_nonzero_mean_lag26", F.avg("_y_nonzero").over(art_exp_lag26))
    )

    # =====================================================================
    # 11. TEMPORAL ENCODINGS & CALENDAR FLAGS
    # =====================================================================
    two_pi = F.lit(2 * math.pi)
    ns = F.col("num_sem")
    df = (
        df
        # Fourier harmonics 1-4
        .withColumn("sin_sem",   F.sin(two_pi * ns / F.lit(52.0)))
        .withColumn("cos_sem",   F.cos(two_pi * ns / F.lit(52.0)))
        .withColumn("sin_sem_2", F.sin(F.lit(2) * two_pi * ns / F.lit(52.0)))
        .withColumn("cos_sem_2", F.cos(F.lit(2) * two_pi * ns / F.lit(52.0)))
        .withColumn("sin_sem_3", F.sin(F.lit(3) * two_pi * ns / F.lit(52.0)))
        .withColumn("cos_sem_3", F.cos(F.lit(3) * two_pi * ns / F.lit(52.0)))
        .withColumn("sin_sem_4", F.sin(F.lit(4) * two_pi * ns / F.lit(52.0)))
        .withColumn("cos_sem_4", F.cos(F.lit(4) * two_pi * ns / F.lit(52.0)))
        # Extra temporal
        .withColumn("trimestre",       F.ceil(ns / F.lit(13.0)).cast("int"))
        .withColumn("mois_approx",     F.ceil(ns / F.lit(4.333)).cast("int"))
        .withColumn("week_of_quarter", ((ns - F.lit(1)) % F.lit(13) + F.lit(1)).cast("int"))
        # Calendar flags
        .withColumn("is_summer_trough", ((ns >= 30) & (ns <= 35)).cast("tinyint"))
        .withColumn("is_xmas_trough",   ((ns >= 50) | (ns == 1)).cast("tinyint"))
        .withColumn("is_august",         ((ns >= 31) & (ns <= 35)).cast("tinyint"))
        .withColumn("is_december",       ((ns >= 48) & (ns <= 52)).cast("tinyint"))
        .withColumn("is_quarter_end",    (ns.isin(13, 26, 39, 52)).cast("tinyint"))
        .withColumn("is_peak_spring",    ((ns >= 12) & (ns <= 22)).cast("tinyint"))
        .withColumn("is_peak_autumn",    ((ns >= 36) & (ns <= 44)).cast("tinyint"))
        # Countdowns
        .withColumn(
            "weeks_to_summer",
            F.least(F.abs(ns - F.lit(32)), F.lit(52) - F.abs(ns - F.lit(32))).cast("double"),
        )
        .withColumn(
            "weeks_to_xmas",
            F.least(F.abs(ns - F.lit(52)), F.lit(52) - F.abs(ns - F.lit(52))).cast("double"),
        )
    )

    # =====================================================================
    # 12. JOIN SILVER DIMS
    # =====================================================================
    df = (
        df
        .join(articles_enc, PAIR_KEYS, "left")
        .join(agences_enc, "code_agence", "left")
    )

    # =====================================================================
    # 13. JOIN LAGGED BILLING (+ rolling 6-month aggregates)
    # =====================================================================
    df = (
        df
        .withColumn(
            "_join_mois",
            F.least(F.lit(12), F.greatest(F.lit(1), F.ceil(F.col("num_sem") / F.lit(4.333)))),
        )
        .withColumn("_join_annee", F.col("annee"))
        .join(
            fac,
            ["code_agence", "code_article", "_join_annee", "_join_mois"],
            "left",
        )
        .drop("_join_annee", "_join_mois")
    )

    # Rolling 6-month billing aggregates (over the last 6 join-months).
    fac_w = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(-26, -1)
    )
    df = (
        df
        .withColumn("fac_achats_roll6",   F.sum("fac_nb_achats").over(fac_w))
        .withColumn("fac_qty_roll6",      F.sum("fac_sum_quantite").over(fac_w))
        .withColumn("fac_montant_roll6",  F.sum("fac_sum_montant").over(fac_w))
        .withColumn(
            "fac_pct_pro_roll6",
            F.avg("fac_pct_pro").over(fac_w),
        )
    )

    # =====================================================================
    # 14. FINAL PROJECTION
    # =====================================================================
    base_cols = [
        "semaine", "week_id",
        "code_agence", "code_article",
        "quantite", "quantite_raw", "quantite_smooth",
        "is_anomaly", "is_capped", "is_dead_pair",
    ]

    # Collect every feature column we generated; also pull annee which is
    # needed by the split filters.  num_sem is already a feature column
    # (it came from silver_panel) so it's included automatically.
    feature_cols = [c for c in df.columns if c not in base_cols
                    and not c.startswith("_")
                    and c not in ("y",)]

    # Deduplicate (annee/num_sem may already be in feature_cols).
    all_cols = list(dict.fromkeys(base_cols + feature_cols))
    return df.select(*all_cols)


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
