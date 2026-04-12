from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import PAIR_KEYS, ANOMALY_ROLL_WINDOW, ANOMALY_MAD_ZSCORE


# ============================================================================
# Helper: parse `YYYY-WW` into annee / num_sem / week_id
# ============================================================================
def _add_time_columns(df, semaine_col="semaine"):
    """Add annee, num_sem and a sortable week_id to a DataFrame."""
    return (
        df.withColumn("annee", F.split(F.col(semaine_col), "-").getItem(0).cast("int"))
          .withColumn("num_sem", F.split(F.col(semaine_col), "-").getItem(1).cast("int"))
          .withColumn("week_id", F.col("annee") * F.lit(100) + F.col("num_sem"))
    )


# ============================================================================
# Helper: deterministic label-encoding via dense_rank
# ============================================================================
def _encode_column(df, src, dst):
    """Append `dst` as a deterministic 0-based dense rank of `src`."""
    if src not in df.columns:
        return df.withColumn(dst, F.lit(-1).cast("int"))
    dim = (
        df.select(src)
        .distinct()
        .withColumn(dst, F.dense_rank().over(Window.orderBy(src)) - F.lit(1))
    )
    return df.join(dim, src, "left")


# ============================================================================
# SILVER: cleaned, parsed, enriched
# ============================================================================

@dp.materialized_view(
    name="silver_ventes",
    comment=(
        "Cleaned sales history: per-pair P99.5 outlier cap, MAD z-score "
        "anomaly clipping (z>3.5), EMA smoothing, and a dead-pair flag."
    ),
    table_properties={"quality": "silver"},
)
@dp.expect_or_drop("semaine_not_null", "semaine IS NOT NULL")
@dp.expect_or_drop("code_agence_not_null", "code_agence IS NOT NULL")
@dp.expect_or_drop("code_article_not_null", "code_article IS NOT NULL")
@dp.expect("quantite_non_negative", "quantite >= 0")
@dp.expect("quantite_reasonable", "quantite <= 25000")
@dp.expect("annee_in_range", "annee BETWEEN 2020 AND 2026")
def silver_ventes():
    raw = (
        spark.read.table("bronze_ventes")
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

    # 2. Cap extreme spikes at max(P99.5, pair_median * 2).
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

    # 3. MAD-based anomaly detection (strictly past window).
    #    z_mad = |x - median| / (1.4826 * MAD)   where MAD = median(|x - median|)
    #    1.4826 makes the MAD consistent with the standard deviation for a
    #    Gaussian, but the z-score is valid for any distribution.
    #    Anomalous values are CLIPPED (not replaced) to median + z * MAD_scaled,
    #    preserving the direction of trend spikes instead of crushing them flat.
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
        # MAD = median(|x_i - median|) over the same window.
        .withColumn(
            "_abs_dev",
            F.abs(F.col("quantite_capped") - F.col("_roll_median")),
        )
        .withColumn(
            "_roll_mad",
            F.expr("percentile_approx(_abs_dev, 0.5)").over(roll_w),
        )
        # Scaled MAD: 1.4826 × MAD. Floor at 1.0 so that constant-value
        # series (MAD=0) still have a usable denominator.
        .withColumn(
            "_mad_scaled",
            F.greatest(F.col("_roll_mad") * F.lit(1.4826), F.lit(1.0)),
        )
        .withColumn(
            "_z_mad",
            F.when(
                F.col("_roll_median").isNotNull(),
                F.col("_abs_dev") / F.col("_mad_scaled"),
            ),
        )
        .withColumn(
            "is_anomaly",
            (
                F.col("_z_mad").isNotNull()
                & (F.col("_z_mad") > F.lit(ANOMALY_MAD_ZSCORE))
                & (F.col("_roll_median") > F.lit(0))
            ).cast("tinyint"),
        )
        # Clip anomalous values instead of replacing: cap at median + threshold × MAD_scaled.
        .withColumn(
            "_anomaly_cap",
            F.col("_roll_median") + F.lit(ANOMALY_MAD_ZSCORE) * F.col("_mad_scaled"),
        )
        .withColumn(
            "quantite",
            F.when(
                F.col("is_anomaly") == 1,
                F.least(F.col("quantite_capped"), F.col("_anomaly_cap")),
            )
            .otherwise(F.col("quantite_capped"))
            .cast("long"),
        )
    )

    # 4. EMA-ish smoothing: 13-week past-only mean.
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

@dp.materialized_view(
    name="silver_articles_encoded",
    comment="Article reference with label-encoded specialite / famille / marque / mdd.",
    table_properties={"quality": "silver"},
)
@dp.expect_or_fail("articles_unique_pair", "code_agence IS NOT NULL AND code_article IS NOT NULL")
def silver_articles_encoded():
    df = spark.read.table("bronze_articles")

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


@dp.materialized_view(
    name="silver_agences_encoded",
    comment="Agency reference with label-encoded region.",
    table_properties={"quality": "silver"},
)
def silver_agences_encoded():
    df = spark.read.table("bronze_agences")
    # The raw column name is `region` in most builds of the hackathon data.
    src = "region" if "region" in df.columns else "ag_region"
    df = df.withColumnRenamed(src, "ag_region")
    df = _encode_column(df, "ag_region", "ag_region_enc")
    return df.select("code_agence", "ag_region_enc").dropDuplicates(["code_agence"])


# ---------------------------------------------------------------------------
# Silver helper: monthly billing aggregated and lagged by 2 months
# ---------------------------------------------------------------------------

@dp.materialized_view(
    name="silver_facturation_lagged",
    comment=(
        "Billing features per (agence, article, join_annee, join_mois). "
        "The _join_* columns are the calendar month shifted by +2 so that a "
        "weekly row matching on those columns only sees billing from >= 8 "
        "weeks ago."
    ),
    table_properties={"quality": "silver"},
)
def silver_facturation_lagged():
    fac = spark.read.table("bronze_facturation")

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
# Silver panel: cleaned train rows + test rows as null-target placeholders
# ---------------------------------------------------------------------------

@dp.materialized_view(
    name="silver_panel",
    comment=(
        "Full weekly panel: cleaned train rows plus test rows with null target. "
        "is_dead_pair is propagated from the train side onto test rows."
    ),
    table_properties={"quality": "silver"},
)
@dp.expect_or_drop("semaine_not_null", "semaine IS NOT NULL")
@dp.expect_or_drop("keys_not_null", "code_agence IS NOT NULL AND code_article IS NOT NULL")
def silver_panel():
    cleaned = spark.read.table("silver_ventes")

    test = (
        spark.read.table("bronze_ventes_test")
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

    # Propagate the dead-pair flag from train rows to every row of the pair.
    pair_w = Window.partitionBy(*PAIR_KEYS)
    return panel.withColumn(
        "is_dead_pair",
        F.max("is_dead_pair").over(pair_w).cast("tinyint"),
    )
