from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Data engineering
# ---------------------------------------------------------------------------
#
# Feature principles:
# - no lag shorter than 26 weeks, because the leaderboard horizon is 26 weeks;
# - article metadata is keyed by (agency, article), never by article alone;
# - billing features use previous-year month/year aggregates only, so they are
#   available for every historical training row, validation row, and hidden-test
#   row without peeking into the target horizon.

KEY_COLS = ["code_agence", "code_article"]
PI = 3.141592653589793


def _parsed_week_columns(df):
    return (
        df.withColumn("annee", F.split(F.col("semaine"), "-").getItem(0).cast("int"))
        .withColumn("num_semaine", F.split(F.col("semaine"), "-").getItem(1).cast("int"))
        .withColumn("week_index", F.col("annee") * F.lit(52) + F.col("num_semaine"))
        .withColumn(
            "target_month",
            F.least(
                F.lit(12),
                F.greatest(F.lit(1), F.ceil(F.col("num_semaine") * F.lit(12.0) / F.lit(52.0))),
            ).cast("int"),
        )
        .withColumn(
            "week_sin",
            F.sin(F.lit(2.0 * PI) * F.col("num_semaine") / F.lit(52.0)),
        )
        .withColumn(
            "week_cos",
            F.cos(F.lit(2.0 * PI) * F.col("num_semaine") / F.lit(52.0)),
        )
    )


def _with_sales_types(df):
    return (
        df.withColumn("code_agence", F.col("code_agence").cast("long"))
        .withColumn("code_article", F.col("code_article").cast("long"))
        .withColumn("quantite", F.greatest(F.col("quantite").cast("double"), F.lit(0.0)))
        .withColumn("is_zero_sale", F.col("quantite") == F.lit(0.0))
        .withColumn("is_positive_sale", F.col("quantite") > F.lit(0.0))
    )


def _with_key_types(df):
    return (
        df.withColumn("code_agence", F.col("code_agence").cast("long"))
        .withColumn("code_article", F.col("code_article").cast("long"))
    )


def _safe_divide(numerator, denominator):
    return F.when(
        denominator.isNotNull() & (denominator != F.lit(0.0)),
        numerator / denominator,
    ).otherwise(F.lit(None).cast("double"))


def _clipped_safe_divide(numerator, denominator, lower=0.0, upper=3.0):
    ratio = _safe_divide(numerator, denominator)
    return F.when(ratio.isNull(), F.lit(None).cast("double")).otherwise(
        F.least(F.lit(upper), F.greatest(F.lit(lower), ratio))
    )


@dp.materialized_view(comment="Training sales data with parsed time features")
def features_train():
    return _with_sales_types(
        _parsed_week_columns(spark.read.table("workspace.default.histo_ventes_train"))
    )


@dp.materialized_view(comment="Test sales data with parsed time features")
def features_test():
    return _with_key_types(
        _parsed_week_columns(spark.read.table("workspace.default.histo_ventes_test"))
    )


@dp.materialized_view(comment="Agency reference features for the LightGBM model")
def agence_features():
    return spark.read.table("workspace.default.donnees_agence").select(
        F.col("code_agence").cast("long").alias("code_agence"),
        F.col("region").cast("string").alias("region"),
        F.col("secteur").cast("string").alias("secteur"),
        F.col("metier").cast("string").alias("metier"),
        F.col("code_postal").cast("string").alias("code_postal"),
        F.col("ville").cast("string").alias("ville"),
        F.col("departement").cast("string").alias("departement"),
        F.col("latitude").cast("double").alias("latitude"),
        F.col("longitude").cast("double").alias("longitude"),
    )


@dp.materialized_view(
    comment="Article reference features keyed by agency and article for the LightGBM model"
)
def article_features():
    return spark.read.table("workspace.default.donnees_articles").select(
        F.col("code_agence").cast("long").alias("code_agence"),
        F.col("code_article").cast("long").alias("code_article"),
        F.col("sous_famille").cast("string").alias("sous_famille"),
        F.col("famille").cast("string").alias("famille"),
        F.col("marque").cast("string").alias("marque"),
        F.col("specialite").cast("string").alias("specialite"),
        F.col("unite_vente").cast("string").alias("unite_vente"),
        F.col("article_mdd").cast("string").alias("article_mdd"),
        F.col("Poids_en_kg").cast("double").alias("poids_en_kg"),
        F.col("Gamme").cast("string").alias("gamme"),
        F.col("code_fournisseur").cast("string").alias("code_fournisseur"),
    )


@dp.materialized_view(comment="Monthly billing features with leakage-safe derived ratios")
def facturation_monthly_features():
    billing = spark.read.table("workspace.default.donnees_facturation").select(
        F.col("code_agence").cast("long").alias("code_agence"),
        F.col("code_article").cast("long").alias("code_article"),
        F.col("annee").cast("int").alias("annee"),
        F.col("mois").cast("int").alias("mois"),
        F.col("nb_achats").cast("double").alias("nb_achats"),
        F.col("sum_quantite").cast("double").alias("sum_quantite"),
        F.col("min_quantite").cast("double").alias("min_quantite"),
        F.col("max_quantite").cast("double").alias("max_quantite"),
        F.col("sum_montant").cast("double").alias("sum_montant"),
        F.col("min_montant").cast("double").alias("min_montant"),
        F.col("max_montant").cast("double").alias("max_montant"),
        F.col("nb_achats_par_professionnels").cast("double").alias(
            "nb_achats_par_professionnels"
        ),
        F.col("nb_achats_par_particuliers").cast("double").alias(
            "nb_achats_par_particuliers"
        ),
        F.col("nb_ventes_magasins").cast("double").alias("nb_ventes_magasins"),
        F.col("nb_ventes_directes").cast("double").alias("nb_ventes_directes"),
        F.col("nb_chantiers").cast("double").alias("nb_chantiers"),
    )

    return (
        billing.withColumn(
            "avg_quantite_par_achat",
            _safe_divide(F.col("sum_quantite"), F.col("nb_achats")),
        )
        .withColumn(
            "avg_montant_par_achat",
            _safe_divide(F.col("sum_montant"), F.col("nb_achats")),
        )
        .withColumn(
            "avg_prix_unitaire",
            _safe_divide(F.col("sum_montant"), F.col("sum_quantite")),
        )
        .withColumn(
            "part_achats_professionnels",
            _safe_divide(
                F.col("nb_achats_par_professionnels"),
                F.col("nb_achats_par_professionnels") + F.col("nb_achats_par_particuliers"),
            ),
        )
        .withColumn(
            "part_ventes_directes",
            _safe_divide(
                F.col("nb_ventes_directes"),
                F.col("nb_ventes_directes") + F.col("nb_ventes_magasins"),
            ),
        )
        .withColumn("has_billing_activity", F.col("nb_achats") > F.lit(0.0))
    )


@dp.materialized_view(
    comment=(
        "Leakage-safe LightGBM feature matrix for train and test rows, enriched "
        "with agency, article, and previous-year billing features"
    )
)
def lightgbm_features():
    train_rows = (
        spark.read.table("features_train")
        .select(
            "semaine",
            "annee",
            "num_semaine",
            "week_index",
            "target_month",
            "week_sin",
            "week_cos",
            "code_agence",
            "code_article",
            "quantite",
        )
        .withColumn("is_test", F.lit(False))
    )

    test_rows = (
        spark.read.table("features_test")
        .select(
            "semaine",
            "annee",
            "num_semaine",
            "week_index",
            "target_month",
            "week_sin",
            "week_cos",
            "code_agence",
            "code_article",
        )
        .withColumn("quantite", F.lit(None).cast("double"))
        .withColumn("is_test", F.lit(True))
    )

    all_rows = train_rows.unionByName(test_rows)
    pair_window = Window.partitionBy(*KEY_COLS).orderBy("week_index")

    lagged = (
        all_rows.withColumn("lag_26", F.lag("quantite", 26).over(pair_window))
        .withColumn("lag_39", F.lag("quantite", 39).over(pair_window))
        .withColumn("lag_52", F.lag("quantite", 52).over(pair_window))
        .withColumn("lag_78", F.lag("quantite", 78).over(pair_window))
        .withColumn("lag_104", F.lag("quantite", 104).over(pair_window))
        .withColumn(
            "rolling_mean_26_52",
            F.avg("quantite").over(pair_window.rowsBetween(-52, -26)),
        )
        .withColumn(
            "rolling_sum_26_52",
            F.sum("quantite").over(pair_window.rowsBetween(-52, -26)),
        )
        .withColumn(
            "rolling_max_26_52",
            F.max("quantite").over(pair_window.rowsBetween(-52, -26)),
        )
        .withColumn(
            "rolling_std_26_52",
            F.stddev("quantite").over(pair_window.rowsBetween(-52, -26)),
        )
        .withColumn(
            "zero_rate_26_52",
            F.avg(F.when(F.col("quantite") == F.lit(0.0), F.lit(1.0)).otherwise(F.lit(0.0))).over(
                pair_window.rowsBetween(-52, -26)
            ),
        )
        .withColumn(
            "rolling_mean_52_104",
            F.avg("quantite").over(pair_window.rowsBetween(-104, -52)),
        )
        .withColumn(
            "rolling_max_52_104",
            F.max("quantite").over(pair_window.rowsBetween(-104, -52)),
        )
        .withColumn(
            "zero_rate_52_104",
            F.avg(F.when(F.col("quantite") == F.lit(0.0), F.lit(1.0)).otherwise(F.lit(0.0))).over(
                pair_window.rowsBetween(-104, -52)
            ),
        )
        .withColumn(
            "pair_mean_to_lag_26",
            F.avg("quantite").over(pair_window.rowsBetween(Window.unboundedPreceding, -26)),
        )
        .withColumn(
            "pair_nonzero_weeks_to_lag_26",
            F.sum(F.when(F.col("quantite") > F.lit(0.0), F.lit(1)).otherwise(F.lit(0))).over(
                pair_window.rowsBetween(Window.unboundedPreceding, -26)
            ),
        )
        .withColumn(
            "pair_zero_rate_to_lag_26",
            F.avg(F.when(F.col("quantite") == F.lit(0.0), F.lit(1.0)).otherwise(F.lit(0.0))).over(
                pair_window.rowsBetween(Window.unboundedPreceding, -26)
            ),
        )
        .withColumn("lag_52_over_lag_104", _safe_divide(F.col("lag_52"), F.col("lag_104")))
        .withColumn("lag_26_over_lag_52", _safe_divide(F.col("lag_26"), F.col("lag_52")))
        .withColumn(
            "recent_over_long_mean",
            _safe_divide(F.col("rolling_mean_26_52"), F.col("rolling_mean_52_104")),
        )
    )

    fact_month = spark.read.table("facturation_monthly_features")
    fact_prev_year_month = fact_month.select(
        *KEY_COLS,
        (F.col("annee") + F.lit(1)).alias("annee"),
        F.col("mois").alias("target_month"),
        F.col("nb_achats").alias("fact_prev_year_month_nb_achats"),
        F.col("sum_quantite").alias("fact_prev_year_month_sum_quantite"),
        F.col("sum_montant").alias("fact_prev_year_month_sum_montant"),
        F.col("max_quantite").alias("fact_prev_year_month_max_quantite"),
        F.col("avg_quantite_par_achat").alias("fact_prev_year_month_avg_quantite_par_achat"),
        F.col("avg_montant_par_achat").alias("fact_prev_year_month_avg_montant_par_achat"),
        F.col("avg_prix_unitaire").alias("fact_prev_year_month_avg_prix_unitaire"),
        F.col("part_achats_professionnels").alias("fact_prev_year_month_part_pro"),
        F.col("part_ventes_directes").alias("fact_prev_year_month_part_direct"),
        F.col("nb_chantiers").alias("fact_prev_year_month_nb_chantiers"),
    )

    fact_prev_year_total = (
        fact_month.groupBy(*(KEY_COLS + ["annee"]))
        .agg(
            F.sum("nb_achats").alias("fact_prev_year_total_nb_achats"),
            F.sum("sum_quantite").alias("fact_prev_year_total_sum_quantite"),
            F.sum("sum_montant").alias("fact_prev_year_total_sum_montant"),
            F.max("max_quantite").alias("fact_prev_year_total_max_quantite"),
            F.sum("nb_chantiers").alias("fact_prev_year_total_nb_chantiers"),
            F.avg("part_achats_professionnels").alias("fact_prev_year_total_part_pro"),
            F.avg("part_ventes_directes").alias("fact_prev_year_total_part_direct"),
        )
        .withColumn(
            "fact_prev_year_total_avg_prix_unitaire",
            _safe_divide(
                F.col("fact_prev_year_total_sum_montant"),
                F.col("fact_prev_year_total_sum_quantite"),
            ),
        )
        .select(
            *KEY_COLS,
            (F.col("annee") + F.lit(1)).alias("annee"),
            "fact_prev_year_total_nb_achats",
            "fact_prev_year_total_sum_quantite",
            "fact_prev_year_total_sum_montant",
            "fact_prev_year_total_max_quantite",
            "fact_prev_year_total_nb_chantiers",
            "fact_prev_year_total_part_pro",
            "fact_prev_year_total_part_direct",
            "fact_prev_year_total_avg_prix_unitaire",
        )
    )

    enriched = (
        lagged.join(spark.read.table("agence_features"), ["code_agence"], "left")
        .join(spark.read.table("article_features"), KEY_COLS, "left")
        .join(fact_prev_year_month, KEY_COLS + ["annee", "target_month"], "left")
        .join(fact_prev_year_total, KEY_COLS + ["annee"], "left")
        .withColumn("has_article_reference", F.col("famille").isNotNull())
        .withColumn("has_prev_year_month_billing", F.col("fact_prev_year_month_nb_achats").isNotNull())
        .withColumn("has_prev_year_total_billing", F.col("fact_prev_year_total_nb_achats").isNotNull())
    )

    agency_week_history = enriched.groupBy("code_agence", "week_index").agg(
        F.sum("quantite").alias("agency_week_volume")
    )
    family_week_history = enriched.groupBy("famille", "week_index").agg(
        F.sum("quantite").alias("family_week_volume")
    )
    agency_family_week_history = enriched.groupBy("code_agence", "famille", "week_index").agg(
        F.sum("quantite").alias("agency_family_week_volume")
    )

    enriched_with_group_lags = (
        enriched.join(
            agency_week_history.select(
                "code_agence",
                (F.col("week_index") + F.lit(26)).alias("week_index"),
                F.col("agency_week_volume").alias("agency_week_volume_lag_26"),
            ),
            ["code_agence", "week_index"],
            "left",
        )
        .join(
            agency_week_history.select(
                "code_agence",
                (F.col("week_index") + F.lit(52)).alias("week_index"),
                F.col("agency_week_volume").alias("agency_week_volume_lag_52"),
            ),
            ["code_agence", "week_index"],
            "left",
        )
        .join(
            family_week_history.select(
                "famille",
                (F.col("week_index") + F.lit(26)).alias("week_index"),
                F.col("family_week_volume").alias("family_week_volume_lag_26"),
            ),
            ["famille", "week_index"],
            "left",
        )
        .join(
            family_week_history.select(
                "famille",
                (F.col("week_index") + F.lit(52)).alias("week_index"),
                F.col("family_week_volume").alias("family_week_volume_lag_52"),
            ),
            ["famille", "week_index"],
            "left",
        )
        .join(
            agency_family_week_history.select(
                "code_agence",
                "famille",
                (F.col("week_index") + F.lit(26)).alias("week_index"),
                F.col("agency_family_week_volume").alias("agency_family_week_volume_lag_26"),
            ),
            ["code_agence", "famille", "week_index"],
            "left",
        )
        .join(
            agency_family_week_history.select(
                "code_agence",
                "famille",
                (F.col("week_index") + F.lit(52)).alias("week_index"),
                F.col("agency_family_week_volume").alias("agency_family_week_volume_lag_52"),
            ),
            ["code_agence", "famille", "week_index"],
            "left",
        )
        .withColumn(
            "agency_volume_trend_26_52",
            _clipped_safe_divide(
                F.col("agency_week_volume_lag_26"),
                F.col("agency_week_volume_lag_52"),
            ),
        )
        .withColumn(
            "family_volume_trend_26_52",
            _clipped_safe_divide(
                F.col("family_week_volume_lag_26"),
                F.col("family_week_volume_lag_52"),
            ),
        )
        .withColumn(
            "agency_family_volume_trend_26_52",
            _clipped_safe_divide(
                F.col("agency_family_week_volume_lag_26"),
                F.col("agency_family_week_volume_lag_52"),
            ),
        )
    )

    return (
        enriched_with_group_lags.withColumn("is_week_1", (F.col("num_semaine") == F.lit(1)).cast("double"))
        .withColumn("is_winter_restart", F.col("num_semaine").between(1, 2).cast("double"))
        .withColumn("is_post_new_year_ramp", F.col("num_semaine").between(3, 5).cast("double"))
        .withColumn("is_spring_high_season", F.col("num_semaine").between(14, 24).cast("double"))
        .withColumn("is_pre_summer_peak", F.col("num_semaine").between(24, 27).cast("double"))
        .withColumn("is_summer_holiday", F.col("num_semaine").between(31, 34).cast("double"))
        .withColumn("is_year_end_holiday", F.col("num_semaine").between(51, 52).cast("double"))
        .withColumn(
            "is_holiday_trough",
            (
                F.col("num_semaine").between(1, 2)
                | F.col("num_semaine").between(31, 34)
                | F.col("num_semaine").between(51, 52)
            ).cast("double"),
        )
        .withColumn(
            "billing_prev_year_month_weekly_rate",
            _safe_divide(F.col("fact_prev_year_month_sum_quantite"), F.lit(4.33)),
        )
        .withColumn(
            "billing_prev_year_total_weekly_rate",
            _safe_divide(F.col("fact_prev_year_total_sum_quantite"), F.lit(52.0)),
        )
        .withColumn(
            "history_strength",
            F.least(
                F.lit(1.0),
                F.coalesce(_safe_divide(F.col("pair_nonzero_weeks_to_lag_26"), F.lit(52.0)), F.lit(0.0)),
            ),
        )
        .withColumn(
            "baseline_trend_factor",
            F.least(
                F.lit(1.60),
                F.greatest(
                    F.lit(0.45),
                    F.coalesce(
                        F.col("recent_over_long_mean"),
                        F.col("lag_26_over_lag_52"),
                        F.col("agency_family_volume_trend_26_52"),
                        F.col("agency_volume_trend_26_52"),
                        F.col("family_volume_trend_26_52"),
                        F.col("lag_52_over_lag_104"),
                        F.lit(1.0),
                    ),
                ),
            ),
        )
        .withColumn(
            "seasonal_baseline_raw",
            F.greatest(
                F.lit(0.0),
                F.coalesce(F.col("lag_52"), F.lit(0.0)) * F.col("baseline_trend_factor") * F.lit(0.42)
                + F.coalesce(F.col("rolling_mean_26_52"), F.lit(0.0)) * F.lit(0.22)
                + F.coalesce(F.col("rolling_mean_52_104"), F.lit(0.0)) * F.lit(0.12)
                + F.coalesce(F.col("pair_mean_to_lag_26"), F.lit(0.0)) * F.lit(0.12)
                + F.coalesce(F.col("billing_prev_year_month_weekly_rate"), F.lit(0.0)) * F.lit(0.08)
                + F.coalesce(F.col("billing_prev_year_total_weekly_rate"), F.lit(0.0)) * F.lit(0.04),
            ),
        )
        .withColumn(
            "calendar_baseline_factor",
            F.when(F.col("is_week_1") == F.lit(1.0), F.lit(0.55))
            .when(F.col("is_winter_restart") == F.lit(1.0), F.lit(0.75))
            .when(F.col("is_summer_holiday") == F.lit(1.0), F.lit(0.72))
            .when(F.col("is_year_end_holiday") == F.lit(1.0), F.lit(0.65))
            .otherwise(F.lit(1.0)),
        )
        .withColumn(
            "sparse_baseline_factor",
            F.when(
                (F.col("pair_zero_rate_to_lag_26") >= F.lit(0.98))
                & (F.coalesce(F.col("lag_52"), F.lit(0.0)) <= F.lit(0.0))
                & (F.coalesce(F.col("rolling_mean_26_52"), F.lit(0.0)) <= F.lit(0.05))
                & (F.coalesce(F.col("fact_prev_year_month_nb_achats"), F.lit(0.0)) <= F.lit(0.0)),
                F.lit(0.03),
            )
            .when(
                (F.col("pair_zero_rate_to_lag_26") >= F.lit(0.95))
                & (F.coalesce(F.col("lag_52"), F.lit(0.0)) <= F.lit(0.0)),
                F.lit(0.18),
            )
            .when(
                (F.col("pair_zero_rate_to_lag_26") >= F.lit(0.85))
                & (F.coalesce(F.col("lag_52"), F.lit(0.0)) <= F.lit(0.0))
                & (F.coalesce(F.col("fact_prev_year_month_nb_achats"), F.lit(0.0)) <= F.lit(0.0)),
                F.lit(0.45),
            )
            .when(
                (F.col("zero_rate_26_52") >= F.lit(0.90))
                & (F.col("seasonal_baseline_raw") <= F.lit(1.0)),
                F.lit(0.70),
            )
            .otherwise(F.lit(1.0)),
        )
        .withColumn(
            "baseline_prediction",
            F.greatest(
                F.lit(0.0),
                F.col("seasonal_baseline_raw") * F.col("calendar_baseline_factor"),
            ),
        )
        .withColumn(
            "sparse_adjusted_baseline",
            F.greatest(
                F.lit(0.0),
                F.col("baseline_prediction") * F.col("sparse_baseline_factor"),
            ),
        )
        .withColumn(
            "baseline_to_recent_ratio",
            _safe_divide(F.col("sparse_adjusted_baseline"), F.col("rolling_mean_26_52")),
        )
        .withColumn(
            "baseline_to_lag52_ratio",
            _safe_divide(F.col("sparse_adjusted_baseline"), F.col("lag_52")),
        )
        .withColumn(
            "baseline_zero_prior",
            (
                (F.col("pair_zero_rate_to_lag_26") >= F.lit(0.95))
                & (F.coalesce(F.col("lag_52"), F.lit(0.0)) <= F.lit(0.0))
                & (F.coalesce(F.col("fact_prev_year_month_nb_achats"), F.lit(0.0)) <= F.lit(0.0))
            ).cast("double"),
        )
    )
