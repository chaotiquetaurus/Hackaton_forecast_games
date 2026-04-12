# Databricks notebook source
# MAGIC %md
# MAGIC # SGDB 2026 — Lakeflow (Delta Live Tables) Pipeline
# MAGIC
# MAGIC Single DLT file that replaces `01_data_cleaning.py` and
# MAGIC `02_feature_engineering.py`. Databricks resolves the DAG automatically
# MAGIC from the `dlt.read()` calls between `@dlt.table` functions.
# MAGIC
# MAGIC **DAG (resolved by DLT):**
# MAGIC
# MAGIC ```
# MAGIC bronze_ventes ───────┐
# MAGIC bronze_ventes_test ──┤
# MAGIC bronze_agences ──────┤
# MAGIC bronze_articles ─────┤
# MAGIC bronze_facturation ──┘
# MAGIC            │
# MAGIC            ▼
# MAGIC silver_ventes  (cleaned, with DQ expectations)
# MAGIC silver_articles_encoded  (label-encoded categoricals)
# MAGIC silver_agences_encoded   (region encoded)
# MAGIC silver_facturation_lagged (monthly agg + 2-month lag)
# MAGIC            │
# MAGIC            ▼
# MAGIC silver_panel  (cleaned train ∪ test placeholders, with time + dead-pair)
# MAGIC            │
# MAGIC            ▼
# MAGIC gold_feature_table
# MAGIC      │         │          │
# MAGIC      ▼         ▼          ▼
# MAGIC gold_train  gold_validation  gold_internal_test  gold_test_features
# MAGIC ```
# MAGIC
# MAGIC **Strict rules observed:**
# MAGIC - `dlt.read("<name>")` for every intermediate table reference.
# MAGIC - `spark.table("workspace.default.*")` only for the five raw source tables
# MAGIC   that the hackathon provisioned.
# MAGIC - Every rolling / expanding window uses `rowsBetween(..., -1)` so the
# MAGIC   current row is never in its own feature window.
# MAGIC - No MLflow, no prints, no CSV writes, no side effects inside DLT functions.
# MAGIC - Expectations live on the silver layer as quality gates.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import math

import dlt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Pair key reused by every windowed feature in the feature layer.
PAIR_KEYS = ["code_agence", "code_article"]


# ============================================================================
# Helper: parse `YYYY-WW` into annee / num_sem / week_id
# ============================================================================
def _add_time_columns(df, semaine_col: str = "semaine"):
    """Add annee, num_sem and a sortable week_id to a DataFrame."""
    return (
        df.withColumn("annee", F.split(F.col(semaine_col), "-").getItem(0).cast("int"))
          .withColumn("num_sem", F.split(F.col(semaine_col), "-").getItem(1).cast("int"))
          .withColumn("week_id", F.col("annee") * F.lit(100) + F.col("num_sem"))
    )


# ============================================================================
# LAYER 1 — BRONZE: Raw ingestion (pass-through from the hackathon tables)
# ============================================================================

def _build_feature_frame(panel, articles_enc, agences_enc, fac, history_end_week_id, split_name):
    """Build features using only targets up to `history_end_week_id`."""
    df = panel.withColumn(
        "y",
        F.when(
            F.col("week_id") <= F.lit(history_end_week_id),
            F.col("quantite").cast("double"),
        ).otherwise(F.lit(None).cast("double")),
    ).withColumn("split_name", F.lit(split_name))

    pair_order = Window.partitionBy(*PAIR_KEYS).orderBy("week_id")
    for n in LAGS_ALL:
        df = df.withColumn(f"lag_{n}", F.lag("y", n).over(pair_order))

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
            F.when(F.col("_mean_prev4").isNull(), F.lit(None).cast("double"))
             .otherwise(
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

    pair_exp = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    df = df.withColumn("_pair_obs_idx", F.row_number().over(pair_order) - F.lit(1))
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

    df = (
        df
        .join(articles_enc, PAIR_KEYS, "left")
        .join(agences_enc, "code_agence", "left")
    )
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

    base_cols = [
        "semaine", "week_id", "split_name",
        "code_agence", "code_article",
        "quantite", "quantite_raw", "quantite_smooth",
        "is_anomaly", "is_capped", "is_dead_pair",
    ]

    for c in FEATURES:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))

    return df.select(*base_cols, *FEATURES)


@dlt.table(
    name="bronze_ventes",
    comment="Raw sales history (histo_ventes_train) — exact passthrough.",
)
def bronze_ventes():
    return spark.table(TBL_TRAIN)


@dlt.table(
    name="bronze_ventes_test",
    comment="Raw test rows to predict (histo_ventes_test) — exact passthrough.",
)
def bronze_ventes_test():
    return spark.table(TBL_TEST)


@dlt.table(
    name="bronze_agences",
    comment="Agency reference data (donnees_agence).",
)
def bronze_agences():
    return spark.table(TBL_AGENCE)


@dlt.table(
    name="bronze_articles",
    comment="Article reference data (donnees_articles) — keyed by (agency, article).",
)
def bronze_articles():
    return spark.table(TBL_ARTICLES)


@dlt.table(
    name="bronze_facturation",
    comment="Monthly billing aggregates (donnees_facturation).",
)
def bronze_facturation():
    return spark.table(TBL_FACTURATION)


# ============================================================================
# LAYER 2 — SILVER: cleaned, parsed, enriched
# ============================================================================

@dlt.table(
    name="silver_ventes",
    comment=(
        "Cleaned sales history: per-pair P99.5 outlier cap, 10×-rolling-median "
        "anomaly imputation, EMA smoothing, and a dead-pair flag."
    ),
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("semaine_not_null", "semaine IS NOT NULL")
@dlt.expect_or_drop("code_agence_not_null", "code_agence IS NOT NULL")
@dlt.expect_or_drop("code_article_not_null", "code_article IS NOT NULL")
@dlt.expect("quantite_non_negative", "quantite >= 0")
@dlt.expect("quantite_reasonable", "quantite <= 25000")
@dlt.expect("annee_in_range", "annee BETWEEN 2020 AND 2026")
def silver_ventes():
    raw = (
        dlt.read("bronze_ventes")
        .transform(_add_time_columns)
        .withColumnRenamed("quantite", "quantite_raw")
    )

    # 1. Dead-pair detection and per-pair quantile stats.
    pair_stats = (
        raw.groupBy(*PAIR_KEYS)
        .agg(
            F.sum("quantite_raw").alias("_pair_sum"),
            F.expr("percentile_approx(quantite_raw, 0.995)").alias("_pair_p995"),
            F.expr("percentile_approx(quantite_raw, 0.5)").alias("_pair_median"),
        )
        .withColumn("is_dead_pair", (F.col("_pair_sum") == 0).cast("tinyint"))
    )

    # 2. Cap extreme spikes at max(P99.5, pair_median × 2).
    #    Dead pairs keep their original zeros.
    capped = (
        raw.join(pair_stats, PAIR_KEYS, "left")
        .withColumn(
            "_cap_value",
            F.when(F.col("is_dead_pair") == 1, F.lit(None))
             .otherwise(
                 F.greatest(F.col("_pair_p995"), F.col("_pair_median") * F.lit(2.0))
             ),
        )
        .withColumn(
            "is_capped",
            (
                F.col("_cap_value").isNotNull()
                & (F.col("quantite_raw") > F.col("_cap_value"))
            ).cast("tinyint"),
        )
        .withColumn(
            "quantite_capped",
            F.when(F.col("is_capped") == 1, F.col("_cap_value").cast("double"))
             .otherwise(F.col("quantite_raw").cast("double")),
        )
    )

    # 3. Rolling-median anomaly detection. The window is strictly past so the
    #    imputation itself cannot leak.
    roll_w = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(-ANOMALY_ROLL_WINDOW, -1)
    )
    with_roll = (
        capped
        .withColumn(
            "_roll_median",
            F.expr("percentile_approx(quantite_capped, 0.5)").over(roll_w),
        )
        .withColumn(
            "is_anomaly",
            (
                F.col("_roll_median").isNotNull()
                & (F.col("_roll_median") > F.lit(0))
                & (
                    F.col("quantite_capped")
                    > F.lit(ANOMALY_MULTIPLIER) * F.col("_roll_median")
                )
            ).cast("tinyint"),
        )
        .withColumn(
            "quantite",
            F.when(F.col("is_anomaly") == 1, F.col("_roll_median"))
             .otherwise(F.col("quantite_capped"))
             .cast("long"),
        )
    )

    # 4. EMA-ish smoothing: 13-week past-only mean used as a denoising feature.
    smooth_w = (
        Window.partitionBy(*PAIR_KEYS)
        .orderBy("week_id")
        .rowsBetween(-13, -1)
    )
    with_smooth = with_roll.withColumn(
        "quantite_smooth", F.avg("quantite").over(smooth_w)
    )

    return with_smooth.select(
        "semaine",
        "annee",
        "num_sem",
        "week_id",
        "code_agence",
        "code_article",
        F.col("quantite_raw").cast("long").alias("quantite_raw"),
        F.col("quantite").cast("long").alias("quantite"),
        F.col("quantite_smooth").cast("double").alias("quantite_smooth"),
        F.col("is_anomaly").cast("tinyint"),
        F.col("is_capped").cast("tinyint"),
        F.col("is_dead_pair").cast("tinyint"),
    )


# ---------------------------------------------------------------------------
# Silver helper dims: categorical encoders
# ---------------------------------------------------------------------------
# We label-encode on the small reference tables (≤ 12K rows) using a
# window-based dense_rank. No driver-side collect, so the encoding is pure
# DLT and survives restart/retry without divergence.

def _encode_column(df, src: str, dst: str):
    """Append `dst` as a deterministic 0-based dense rank of `src`."""
    if src not in df.columns:
        return df.withColumn(dst, F.lit(-1).cast("int"))
    dim = (
        df.select(src)
        .distinct()
        .withColumn(dst, F.dense_rank().over(Window.orderBy(src)) - F.lit(1))
    )
    return df.join(dim, src, "left")


@dlt.table(
    name="silver_articles_encoded",
    comment="Article reference with label-encoded specialite / famille / marque / mdd.",
    table_properties={"quality": "silver"},
)
@dlt.expect_or_fail("articles_unique_pair", "code_agence IS NOT NULL AND code_article IS NOT NULL")
def silver_articles_encoded():
    df = dlt.read("bronze_articles")

    mapping = [
        ("specialite", "art_specialite_enc"),
        ("famille", "art_famille_enc"),
        ("marque", "art_marque_enc"),
        ("article_mdd", "art_mdd_enc"),
    ]
    for src, dst in mapping:
        df = _encode_column(df, src, dst)

    keep = ["code_agence", "code_article"] + [dst for _, dst in mapping]
    return df.select(*keep).dropDuplicates(["code_agence", "code_article"])


@dlt.table(
    name="silver_agences_encoded",
    comment="Agency reference with label-encoded region.",
    table_properties={"quality": "silver"},
)
def silver_agences_encoded():
    df = dlt.read("bronze_agences")
    # The raw column name is `region` in most builds of the hackathon data.
    src = "region" if "region" in df.columns else "ag_region"
    df = df.withColumnRenamed(src, "ag_region")
    df = _encode_column(df, "ag_region", "ag_region_enc")
    return df.select("code_agence", "ag_region_enc").dropDuplicates(["code_agence"])


# ---------------------------------------------------------------------------
# Silver helper: monthly billing aggregated and lagged by 2 months
# ---------------------------------------------------------------------------

@dlt.table(
    name="silver_facturation_lagged",
    comment=(
        "Billing features per (agence, article, join_annee, join_mois). "
        "The _join_* columns are the calendar month shifted by +2 so that a "
        "weekly row matching on those columns only sees billing from ≥ 8 "
        "weeks ago."
    ),
    table_properties={"quality": "silver"},
)
def silver_facturation_lagged():
    fac = dlt.read("bronze_facturation")

    def _pick(*candidates, default=None):
        for c in candidates:
            if c in fac.columns:
                return F.col(c)
        return F.lit(default)

    year_col = _pick("annee", "year")
    month_col = _pick("mois", "month")

    monthly = (
        fac
        .withColumn("_annee", year_col.cast("int"))
        .withColumn("_mois", month_col.cast("int"))
        .groupBy("code_agence", "code_article", "_annee", "_mois")
        .agg(
            F.sum(_pick("sum_montant", default=0.0)).alias("_sum_montant"),
            F.sum(_pick("sum_quantite", default=0.0)).alias("_sum_quantite"),
            F.sum(_pick("nb_achats", default=0.0)).alias("fac_nb_achats"),
            F.sum(_pick("nb_achats_par_professionnels", default=0.0)).alias("_nb_pro"),
            F.sum(_pick("nb_chantiers", default=0.0)).alias("fac_nb_chantiers"),
        )
        .withColumn(
            "fac_prix_unit",
            F.when(
                (F.col("_sum_quantite").isNull()) | (F.col("_sum_quantite") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("_sum_montant") / F.col("_sum_quantite")),
        )
        .withColumn(
            "fac_pct_pro",
            F.when(
                (F.col("fac_nb_achats").isNull()) | (F.col("fac_nb_achats") == 0),
                F.lit(None).cast("double"),
            ).otherwise(F.col("_nb_pro") / F.col("fac_nb_achats")),
        )
    )

    # Shift forward 2 months for the join key.
    return (
        monthly
        .withColumn("_shifted_mois", F.col("_mois") + F.lit(2))
        .withColumn(
            "_join_annee",
            F.when(F.col("_shifted_mois") > F.lit(12), F.col("_annee") + F.lit(1))
             .otherwise(F.col("_annee")),
        )
        .withColumn(
            "_join_mois",
            F.when(F.col("_shifted_mois") > F.lit(12), F.col("_shifted_mois") - F.lit(12))
             .otherwise(F.col("_shifted_mois")),
        )
        .select(
            "code_agence",
            "code_article",
            "_join_annee",
            "_join_mois",
            "fac_prix_unit",
            "fac_pct_pro",
            F.col("fac_nb_chantiers").cast("double"),
            F.col("fac_nb_achats").cast("double"),
        )
    )


# ---------------------------------------------------------------------------
# Silver panel: cleaned train rows ∪ test rows as null-target placeholders
# ---------------------------------------------------------------------------

@dlt.table(
    name="silver_panel",
    comment=(
        "Full weekly panel: cleaned train rows plus test rows with null target. "
        "is_dead_pair is propagated from the train side onto test rows so that "
        "inference can shortcut dead pairs without a separate lookup."
    ),
    table_properties={"quality": "silver"},
)
@dlt.expect_or_drop("semaine_not_null", "semaine IS NOT NULL")
@dlt.expect_or_drop("keys_not_null", "code_agence IS NOT NULL AND code_article IS NOT NULL")
def silver_panel():
    cleaned = dlt.read("silver_ventes")

    test = (
        dlt.read("bronze_ventes_test")
        .transform(_add_time_columns)
        .withColumn("quantite_raw", F.lit(None).cast("long"))
        .withColumn("quantite", F.lit(None).cast("long"))
        .withColumn("quantite_smooth", F.lit(None).cast("double"))
        .withColumn("is_anomaly", F.lit(0).cast("tinyint"))
        .withColumn("is_capped", F.lit(0).cast("tinyint"))
        .withColumn("is_dead_pair", F.lit(0).cast("tinyint"))
        .select(*cleaned.columns)
    )

    panel = cleaned.unionByName(test)

    # Propagate the dead-pair flag from train rows to every row of the pair
    # (including test). Max over a window partitioned by pair does it in one
    # shuffle without an extra join.
    pair_w = Window.partitionBy(*PAIR_KEYS)
    return panel.withColumn(
        "is_dead_pair",
        F.max("is_dead_pair").over(pair_w).cast("tinyint"),
    )


# ============================================================================
# LAYER 3 — GOLD: feature table + temporal splits
# ============================================================================

@dlt.table(
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
    panel = dlt.read("silver_panel")
    articles_enc = dlt.read("silver_articles_encoded")
    agences_enc = dlt.read("silver_agences_encoded")
    fac = dlt.read("silver_facturation_lagged")

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

    # Fill in any FEATURES missing from `df` as nulls so downstream consumers
    # can rely on a stable schema even if one feature fails to compute.
    for c in FEATURES:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("double"))

    # `annee` and `num_sem` live in FEATURES (they're real inputs to the
    # model), so we keep them there and leave them out of base_cols to avoid
    # a duplicate column in the final select. They're still present in the
    # resulting Delta table — just exactly once.
    return df.select(*base_cols, *FEATURES)


# ---------------------------------------------------------------------------
# Gold splits
# ---------------------------------------------------------------------------

@dlt.table(
    name="gold_train",
    comment="Training split: weeks up to 2024-W26.",
    table_properties={"quality": "gold"},
)
@dlt.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_train():
    return dlt.read("gold_feature_table").filter(F.col("split_name") == "train")


@dlt.table(
    name="gold_validation",
    comment="Validation split: 2024-W27 .. 2024-W52, with masked-horizon features.",
    table_properties={"quality": "gold"},
)
@dlt.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_validation():
    return dlt.read("gold_feature_table").filter(F.col("split_name") == "validation")


@dlt.table(
    name="gold_internal_test",
    comment="Labelled internal test split: 2025-W01 .. 2025-W26, never used for training.",
    table_properties={"quality": "gold"},
)
@dlt.expect_or_drop("quantite_not_null", "quantite IS NOT NULL")
def gold_internal_test():
    return dlt.read("gold_feature_table").filter(F.col("split_name") == "internal_test")


@dlt.table(
    name="gold_test_features",
    comment=(
        "Final-inference feature rows for the leaderboard period "
        "quantite is null — consumed by the inference notebook to score the "
        "(2025-W27 .. 2025-W52)."
    ),
    table_properties={"quality": "gold"},
)
@dlt.expect_or_fail("quantite_is_null", "quantite IS NULL")
def gold_test_features():
    return dlt.read("gold_feature_table").filter(F.col("split_name") == "final_inference")
