from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Evaluation for the single production model
# ---------------------------------------------------------------------------
#
# These views test the LightGBM residualized hurdle model in depth. They do not choose
# between model families; their purpose is to explain where this precise model
# wins or fails so the next feature/model iteration is obvious.

VALIDATION_YEAR = 2025
VALIDATION_WEEK_MIN = 1
VALIDATION_WEEK_MAX = 26
PREVIOUS_YEAR = VALIDATION_YEAR - 1

ID_COLS = ["semaine", "code_agence", "code_article"]
KEY_COLS = ["code_agence", "code_article"]

FEATURES_TO_MONITOR = [
    "lag_26",
    "lag_52",
    "lag_104",
    "rolling_mean_26_52",
    "zero_rate_26_52",
    "pair_mean_to_lag_26",
    "agency_week_volume_lag_52",
    "family_week_volume_lag_52",
    "agency_family_week_volume_lag_52",
    "agency_family_volume_trend_26_52",
    "baseline_prediction",
    "sparse_adjusted_baseline",
    "baseline_zero_prior",
    "calendar_baseline_factor",
    "sparse_baseline_factor",
    "fact_prev_year_month_nb_achats",
    "fact_prev_year_total_sum_quantite",
    "famille",
    "specialite",
    "marque",
    "region",
]


def _safe_divide(numerator, denominator):
    return F.when(
        denominator.isNotNull() & (denominator != F.lit(0.0)),
        numerator / denominator,
    ).otherwise(F.lit(None).cast("double"))


def _validation_predictions():
    return (
        spark.read.table("validation_predictions")
        .withColumn("annee", F.split(F.col("semaine"), "-").getItem(0).cast("int"))
        .withColumn("num_semaine", F.split(F.col("semaine"), "-").getItem(1).cast("int"))
        .withColumn("prediction", F.greatest(F.col("prediction").cast("double"), F.lit(0.0)))
        .withColumn(
            "model_baseline_prediction",
            F.greatest(F.col("model_baseline_prediction").cast("double"), F.lit(0.0)),
        )
        .withColumn("raw_prediction", F.greatest(F.col("raw_prediction").cast("double"), F.lit(0.0)))
        .withColumn("zero_guard_multiplier", F.col("zero_guard_multiplier").cast("double"))
        .withColumn("quantite", F.greatest(F.col("quantite").cast("double"), F.lit(0.0)))
        .withColumn("absolute_error", F.abs(F.col("quantite") - F.col("prediction")))
        .withColumn("signed_error", F.col("prediction") - F.col("quantite"))
        .withColumn("actual_is_positive", F.col("quantite") > F.lit(0.0))
        .withColumn("predicted_is_positive", F.col("prediction") > F.lit(0.01))
    )


def _validation_features():
    return spark.read.table("lightgbm_features").filter(
        (F.col("is_test") == F.lit(False))
        & (F.col("annee") == F.lit(VALIDATION_YEAR))
        & (F.col("num_semaine").between(VALIDATION_WEEK_MIN, VALIDATION_WEEK_MAX))
    )


def _validation_with_features():
    feature_columns = [
        *ID_COLS,
        "famille",
        "sous_famille",
        "specialite",
        "marque",
        "region",
        "departement",
        "pair_zero_rate_to_lag_26",
        "pair_nonzero_weeks_to_lag_26",
        "rolling_mean_26_52",
        "zero_rate_26_52",
        "lag_26",
        "lag_52",
        "agency_week_volume_lag_26",
        "agency_week_volume_lag_52",
        "family_week_volume_lag_26",
        "family_week_volume_lag_52",
        "agency_family_week_volume_lag_26",
        "agency_family_week_volume_lag_52",
        "agency_family_volume_trend_26_52",
        "sparse_adjusted_baseline",
        "seasonal_baseline_raw",
        "calendar_baseline_factor",
        "sparse_baseline_factor",
        "baseline_zero_prior",
        "history_strength",
        "is_week_1",
        "is_holiday_trough",
        "fact_prev_year_month_nb_achats",
        "fact_prev_year_total_sum_quantite",
    ]
    return _validation_predictions().join(
        _validation_features().select(*feature_columns),
        ID_COLS,
        "left",
    )


def _wape(actual_col="quantite", prediction_col="prediction"):
    return _safe_divide(
        F.sum(F.abs(F.col(actual_col) - F.col(prediction_col))),
        F.sum(F.abs(F.col(actual_col))),
    )


def _metric_aggregations():
    actual_volume = F.sum(F.abs(F.col("quantite")))
    predicted_volume = F.sum(F.col("prediction"))
    absolute_error = F.sum(F.col("absolute_error"))
    signed_error = F.sum(F.col("signed_error"))

    return [
        _wape().alias("wape"),
        F.count("*").alias("n_rows"),
        F.countDistinct(*KEY_COLS).alias("n_pairs"),
        actual_volume.alias("actual_volume"),
        predicted_volume.alias("predicted_volume"),
        absolute_error.alias("absolute_error"),
        signed_error.alias("signed_error"),
        _safe_divide(signed_error, actual_volume).alias("bias_over_actual_volume"),
        F.avg(F.col("absolute_error")).alias("mae"),
        F.avg(F.col("signed_error")).alias("mean_bias"),
        F.sum(
            F.when(F.col("quantite") == F.lit(0.0), F.col("prediction")).otherwise(F.lit(0.0))
        ).alias("predicted_volume_on_actual_zeros"),
        F.sum(
            F.when(
                (F.col("quantite") > F.lit(0.0)) & (F.col("prediction") <= F.lit(0.01)),
                F.col("quantite"),
            ).otherwise(F.lit(0.0))
        ).alias("missed_actual_volume_from_false_zeros"),
        F.sum("model_baseline_prediction").alias("model_baseline_volume"),
        F.sum("raw_prediction").alias("raw_prediction_volume_before_zero_guard"),
        F.avg("zero_guard_multiplier").alias("avg_zero_guard_multiplier"),
    ]


def _with_actual_volume_segment(df):
    return df.withColumn(
        "actual_volume_segment",
        F.when(F.col("quantite") == F.lit(0.0), F.lit("00_zero"))
        .when(F.col("quantite") <= F.lit(5.0), F.lit("01_low_1_to_5"))
        .when(F.col("quantite") <= F.lit(20.0), F.lit("02_medium_6_to_20"))
        .when(F.col("quantite") <= F.lit(100.0), F.lit("03_high_21_to_100"))
        .otherwise(F.lit("04_extreme_gt_100"))
    )


def _with_probability_bucket(df):
    return df.withColumn(
        "sale_probability_bucket",
        F.when(F.col("sale_probability") < F.lit(0.05), F.lit("00_lt_05pct"))
        .when(F.col("sale_probability") < F.lit(0.15), F.lit("01_05_15pct"))
        .when(F.col("sale_probability") < F.lit(0.35), F.lit("02_15_35pct"))
        .when(F.col("sale_probability") < F.lit(0.65), F.lit("03_35_65pct"))
        .otherwise(F.lit("04_ge_65pct"))
    )


def _with_intermittency_segment(df):
    return df.withColumn(
        "intermittency_segment",
        F.when(F.col("pair_zero_rate_to_lag_26") >= F.lit(0.95), F.lit("00_very_sparse"))
        .when(F.col("pair_zero_rate_to_lag_26") >= F.lit(0.75), F.lit("01_sparse"))
        .when(F.col("pair_zero_rate_to_lag_26") >= F.lit(0.50), F.lit("02_mixed"))
        .otherwise(F.lit("03_regular"))
    )


@dp.materialized_view(comment="Overall validation WAPE for the LightGBM residualized hurdle model")
def wape_score():
    return _validation_predictions().groupBy("model_version").agg(
        _wape().alias("wape"),
        F.count("*").alias("n_validation_samples"),
        F.countDistinct(*KEY_COLS).alias("n_validation_pairs"),
        F.sum("quantite").alias("actual_volume"),
        F.sum("prediction").alias("predicted_volume"),
        F.sum("model_baseline_prediction").alias("model_baseline_volume"),
        F.sum("raw_prediction").alias("raw_prediction_volume_before_zero_guard"),
        F.avg("sale_probability").alias("avg_sale_probability"),
        F.avg("zero_guard_multiplier").alias("avg_zero_guard_multiplier"),
    )


@dp.materialized_view(comment="Core validation diagnostics for the LightGBM residualized hurdle model")
def validation_model_diagnostics():
    return _validation_predictions().groupBy("model_version").agg(*_metric_aggregations())


@dp.materialized_view(comment="Occurrence diagnostics for the zero-inflated hurdle layer")
def validation_occurrence_diagnostics():
    df = _validation_predictions()
    return df.groupBy("model_version").agg(
        F.count("*").alias("n_rows"),
        F.sum(F.when(F.col("actual_is_positive"), F.lit(1)).otherwise(F.lit(0))).alias(
            "actual_positive_rows"
        ),
        F.sum(F.when(F.col("predicted_is_positive"), F.lit(1)).otherwise(F.lit(0))).alias(
            "predicted_positive_rows"
        ),
        F.sum(
            F.when(F.col("actual_is_positive") & F.col("predicted_is_positive"), F.lit(1)).otherwise(
                F.lit(0)
            )
        ).alias("true_positive_rows"),
        F.sum(
            F.when(~F.col("actual_is_positive") & F.col("predicted_is_positive"), F.lit(1)).otherwise(
                F.lit(0)
            )
        ).alias("false_positive_zero_rows"),
        F.sum(
            F.when(F.col("actual_is_positive") & ~F.col("predicted_is_positive"), F.lit(1)).otherwise(
                F.lit(0)
            )
        ).alias("false_zero_rows"),
        F.sum(
            F.when(~F.col("actual_is_positive"), F.col("prediction")).otherwise(F.lit(0.0))
        ).alias("predicted_volume_on_actual_zeros"),
        F.sum(
            F.when(
                F.col("actual_is_positive") & ~F.col("predicted_is_positive"),
                F.col("quantite"),
            ).otherwise(F.lit(0.0))
        ).alias("missed_actual_volume_from_false_zeros"),
        F.avg("sale_probability").alias("avg_sale_probability"),
    )


@dp.materialized_view(comment="Validation WAPE by week")
def validation_wape_by_week():
    return (
        _validation_predictions()
        .groupBy("model_version", "annee", "num_semaine")
        .agg(*_metric_aggregations())
        .orderBy("num_semaine")
    )


@dp.materialized_view(comment="Validation WAPE by agency")
def validation_wape_by_agency():
    return (
        _validation_with_features()
        .groupBy("model_version", "code_agence", "region", "departement")
        .agg(*_metric_aggregations())
        .orderBy("wape")
    )


@dp.materialized_view(comment="Validation WAPE by article family and specialty")
def validation_wape_by_article_segment():
    return (
        _validation_with_features()
        .groupBy("model_version", "famille", "sous_famille", "specialite")
        .agg(*_metric_aggregations())
        .filter(F.col("n_rows") >= F.lit(50))
        .orderBy(F.desc("actual_volume"))
    )


@dp.materialized_view(comment="Validation WAPE by actual quantity bucket")
def validation_wape_by_actual_volume_segment():
    return (
        _with_actual_volume_segment(_validation_with_features())
        .groupBy("model_version", "actual_volume_segment")
        .agg(*_metric_aggregations())
        .orderBy("actual_volume_segment")
    )


@dp.materialized_view(comment="Validation WAPE by historical intermittency bucket")
def validation_wape_by_intermittency_segment():
    return (
        _with_intermittency_segment(_validation_with_features())
        .groupBy("model_version", "intermittency_segment")
        .agg(
            *_metric_aggregations(),
            F.avg("pair_zero_rate_to_lag_26").alias("avg_pair_zero_rate_to_lag_26"),
            F.avg("pair_nonzero_weeks_to_lag_26").alias("avg_pair_nonzero_weeks_to_lag_26"),
        )
        .orderBy("intermittency_segment")
    )


@dp.materialized_view(comment="Validation calibration by predicted sale-probability bucket")
def validation_probability_calibration():
    return (
        _with_probability_bucket(_validation_with_features())
        .groupBy("model_version", "sale_probability_bucket")
        .agg(
            *_metric_aggregations(),
            F.avg(F.when(F.col("quantite") > F.lit(0.0), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "actual_positive_rate"
            ),
            F.avg("sale_probability").alias("avg_sale_probability"),
            F.avg("direct_prediction").alias("avg_direct_prediction"),
            F.avg("hurdle_prediction").alias("avg_hurdle_prediction"),
            F.avg("model_baseline_prediction").alias("avg_model_baseline_prediction"),
            F.avg("raw_prediction").alias("avg_raw_prediction_before_zero_guard"),
        )
        .orderBy("sale_probability_bucket")
    )


@dp.materialized_view(comment="Actual-zero overprediction diagnostics by sparse history and probability")
def validation_zero_overprediction_by_sparse_probability():
    return (
        _with_probability_bucket(_with_intermittency_segment(_validation_with_features()))
        .filter(F.col("quantite") == F.lit(0.0))
        .groupBy("model_version", "intermittency_segment", "sale_probability_bucket")
        .agg(
            F.count("*").alias("n_actual_zero_rows"),
            F.sum("prediction").alias("predicted_volume_on_actual_zeros"),
            F.sum("raw_prediction").alias("raw_volume_before_zero_guard_on_actual_zeros"),
            F.avg("prediction").alias("avg_prediction_on_actual_zeros"),
            F.avg("raw_prediction").alias("avg_raw_prediction_before_zero_guard"),
            F.avg("model_baseline_prediction").alias("avg_model_baseline_prediction"),
            F.avg("direct_prediction").alias("avg_direct_prediction"),
            F.avg("hurdle_prediction").alias("avg_hurdle_prediction"),
            F.avg("sale_probability").alias("avg_sale_probability"),
            F.avg("zero_guard_multiplier").alias("avg_zero_guard_multiplier"),
            F.avg("pair_zero_rate_to_lag_26").alias("avg_pair_zero_rate_to_lag_26"),
            F.avg("sparse_baseline_factor").alias("avg_sparse_baseline_factor"),
            F.avg("baseline_zero_prior").alias("avg_baseline_zero_prior"),
        )
        .orderBy("intermittency_segment", "sale_probability_bucket")
    )


@dp.materialized_view(comment="Largest validation errors for manual inspection")
def validation_largest_errors():
    ranked = _validation_with_features().withColumn(
        "error_rank",
        F.row_number().over(Window.orderBy(F.desc("absolute_error"))),
    )
    return (
        ranked.filter(F.col("error_rank") <= F.lit(100))
        .select(
            "error_rank",
            "semaine",
            "code_agence",
            "code_article",
            "famille",
            "sous_famille",
            "specialite",
            "region",
            "quantite",
            "prediction",
            "raw_prediction",
            "model_baseline_prediction",
            "direct_prediction",
            "hurdle_prediction",
            "sale_probability",
            "zero_guard_multiplier",
            "absolute_error",
            "signed_error",
            "lag_26",
            "lag_52",
            "rolling_mean_26_52",
            "zero_rate_26_52",
            "agency_week_volume_lag_52",
            "family_week_volume_lag_52",
            "agency_family_week_volume_lag_52",
            "agency_family_volume_trend_26_52",
            "sparse_adjusted_baseline",
            "seasonal_baseline_raw",
            "calendar_baseline_factor",
            "sparse_baseline_factor",
            "baseline_zero_prior",
            "fact_prev_year_month_nb_achats",
            "fact_prev_year_total_sum_quantite",
        )
        .orderBy("error_rank")
    )


@dp.materialized_view(comment="Feature missing-rate checks on validation and hidden-test rows")
def model_feature_quality_checks():
    monitored = spark.read.table("lightgbm_features").filter(
        (F.col("is_test") == F.lit(True))
        | (
            (F.col("is_test") == F.lit(False))
            & (F.col("annee") == F.lit(VALIDATION_YEAR))
            & (F.col("num_semaine").between(VALIDATION_WEEK_MIN, VALIDATION_WEEK_MAX))
        )
    )

    return monitored.groupBy("is_test").agg(
        F.count("*").alias("n_rows"),
        *[
            F.avg(F.when(F.col(feature).isNull(), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                f"missing_rate__{feature}"
            )
            for feature in FEATURES_TO_MONITOR
        ],
    )


@dp.materialized_view(comment="Validation split integrity checks")
def validation_split_checks():
    validation = _validation_predictions()
    summary = validation.agg(
        F.count("*").alias("n_validation_rows"),
        F.countDistinct(*KEY_COLS).alias("n_validation_pairs"),
        F.sum("quantite").alias("validation_actual_volume"),
        F.min("annee").alias("min_validation_year"),
        F.max("annee").alias("max_validation_year"),
        F.min("num_semaine").alias("min_validation_week"),
        F.max("num_semaine").alias("max_validation_week"),
    )

    return summary.select(
        F.explode(
            F.array(
                F.struct(
                    F.lit("validation_rows_present").alias("check_name"),
                    F.when(F.col("n_validation_rows") > F.lit(0), F.lit("PASS")).otherwise(
                        F.lit("FAIL")
                    ).alias("status"),
                    F.concat(
                        F.lit("n_validation_rows="),
                        F.col("n_validation_rows").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("validation_denominator_nonzero").alias("check_name"),
                    F.when(F.col("validation_actual_volume") > F.lit(0.0), F.lit("PASS")).otherwise(
                        F.lit("FAIL")
                    ).alias("status"),
                    F.concat(
                        F.lit("validation_actual_volume="),
                        F.col("validation_actual_volume").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("validation_period_is_2025_h1").alias("check_name"),
                    F.when(
                        (F.col("min_validation_year") == F.lit(VALIDATION_YEAR))
                        & (F.col("max_validation_year") == F.lit(VALIDATION_YEAR))
                        & (F.col("min_validation_week") >= F.lit(VALIDATION_WEEK_MIN))
                        & (F.col("max_validation_week") <= F.lit(VALIDATION_WEEK_MAX)),
                        F.lit("PASS"),
                    )
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("year_range="),
                        F.col("min_validation_year").cast("string"),
                        F.lit("-"),
                        F.col("max_validation_year").cast("string"),
                        F.lit(", week_range="),
                        F.col("min_validation_week").cast("string"),
                        F.lit("-"),
                        F.col("max_validation_week").cast("string"),
                    ).alias("details"),
                ),
            )
        ).alias("check")
    ).select("check.*")


@dp.materialized_view(comment="Hidden-test prediction table sanity checks")
def submission_prediction_checks():
    test_keys = spark.read.table("features_test").select(*ID_COLS)
    predictions = spark.read.table("predictions").select(
        *ID_COLS,
        F.col("predicted_quantite").cast("double").alias("predicted_quantite"),
    )

    test_summary = test_keys.agg(
        F.count("*").alias("n_test_rows"),
        F.countDistinct(*ID_COLS).alias("n_distinct_test_keys"),
    ).withColumn("n_duplicate_test_rows", F.col("n_test_rows") - F.col("n_distinct_test_keys"))

    prediction_summary = predictions.agg(
        F.count("*").alias("n_prediction_rows"),
        F.countDistinct(*ID_COLS).alias("n_distinct_prediction_keys"),
        F.sum(
            F.when(
                F.col("semaine").isNull()
                | F.col("code_agence").isNull()
                | F.col("code_article").isNull(),
                F.lit(1),
            ).otherwise(F.lit(0))
        ).alias("n_prediction_rows_with_null_key"),
        F.sum(F.when(F.col("predicted_quantite").isNull(), F.lit(1)).otherwise(F.lit(0))).alias(
            "n_null_predictions"
        ),
        F.sum(
            F.when(F.col("predicted_quantite") < F.lit(0.0), F.lit(1)).otherwise(F.lit(0))
        ).alias("n_negative_predictions"),
    ).withColumn(
        "n_duplicate_prediction_rows",
        F.col("n_prediction_rows") - F.col("n_distinct_prediction_keys"),
    )

    coverage = (
        test_keys.distinct()
        .withColumn("_in_test", F.lit(1))
        .join(
            predictions.select(*ID_COLS).distinct().withColumn("_in_predictions", F.lit(1)),
            ID_COLS,
            "full",
        )
        .agg(
            F.sum(
                F.when(
                    F.col("_in_test").isNotNull() & F.col("_in_predictions").isNull(),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("n_missing_prediction_keys"),
            F.sum(
                F.when(
                    F.col("_in_test").isNull() & F.col("_in_predictions").isNotNull(),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ).alias("n_extra_prediction_keys"),
        )
    )

    summary = test_summary.crossJoin(prediction_summary).crossJoin(coverage)
    return summary.select(
        F.explode(
            F.array(
                F.struct(
                    F.lit("row_count_matches_test").alias("check_name"),
                    F.when(F.col("n_prediction_rows") == F.col("n_test_rows"), F.lit("PASS"))
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("predictions="),
                        F.col("n_prediction_rows").cast("string"),
                        F.lit(", test_rows="),
                        F.col("n_test_rows").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("unique_submission_keys").alias("check_name"),
                    F.when(F.col("n_duplicate_prediction_rows") == F.lit(0), F.lit("PASS"))
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("duplicate_prediction_rows="),
                        F.col("n_duplicate_prediction_rows").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("all_test_keys_predicted").alias("check_name"),
                    F.when(F.col("n_missing_prediction_keys") == F.lit(0), F.lit("PASS"))
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("missing_prediction_keys="),
                        F.col("n_missing_prediction_keys").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("no_extra_prediction_keys").alias("check_name"),
                    F.when(F.col("n_extra_prediction_keys") == F.lit(0), F.lit("PASS"))
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("extra_prediction_keys="),
                        F.col("n_extra_prediction_keys").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("no_null_predictions").alias("check_name"),
                    F.when(F.col("n_null_predictions") == F.lit(0), F.lit("PASS"))
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("null_predictions="),
                        F.col("n_null_predictions").cast("string"),
                    ).alias("details"),
                ),
                F.struct(
                    F.lit("no_negative_predictions").alias("check_name"),
                    F.when(F.col("n_negative_predictions") == F.lit(0), F.lit("PASS"))
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("negative_predictions="),
                        F.col("n_negative_predictions").cast("string"),
                    ).alias("details"),
                ),
            )
        ).alias("check")
    ).select("check.*")


@dp.materialized_view(comment="Hidden-test prediction profile by week")
def test_prediction_profile_by_week():
    test_rows = spark.read.table("features_test").select(*ID_COLS, "annee", "num_semaine")
    predictions = spark.read.table("predictions").select(
        *ID_COLS,
        F.col("predicted_quantite").cast("double").alias("predicted_quantite"),
    )

    previous_year_week_volume = (
        spark.read.table("features_train")
        .filter(F.col("annee") == F.lit(PREVIOUS_YEAR))
        .groupBy("num_semaine")
        .agg(F.sum("quantite").alias("actual_volume_same_week_2024"))
    )

    return (
        test_rows.join(predictions, ID_COLS, "left")
        .groupBy("annee", "num_semaine")
        .agg(
            F.count("*").alias("n_prediction_rows"),
            F.sum("predicted_quantite").alias("predicted_volume"),
            F.avg("predicted_quantite").alias("avg_prediction"),
            F.expr("percentile_approx(predicted_quantite, 0.5, 100)").alias(
                "median_prediction"
            ),
            F.avg(
                F.when(F.col("predicted_quantite") <= F.lit(0.01), F.lit(1.0)).otherwise(
                    F.lit(0.0)
                )
            ).alias("prediction_zero_rate"),
        )
        .join(previous_year_week_volume, "num_semaine", "left")
        .withColumn(
            "predicted_to_2024_volume_ratio",
            _safe_divide(F.col("predicted_volume"), F.col("actual_volume_same_week_2024")),
        )
        .orderBy("annee", "num_semaine")
    )
