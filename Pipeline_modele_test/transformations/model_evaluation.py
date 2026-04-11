from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
#
# The hidden leaderboard evaluates 2025-W27 -> 2025-W52, so these validation
# tests use the available 2025-W01 -> 2025-W26 period and train only on rows
# strictly before 2025. This keeps the diagnostics aligned with the hackathon
# time-series constraints and avoids leakage from the validation horizon.

VALIDATION_YEAR = 2025
VALIDATION_WEEK_MIN = 1
VALIDATION_WEEK_MAX = 26
PREVIOUS_YEAR = VALIDATION_YEAR - 1
TREND_REFERENCE_YEAR = VALIDATION_YEAR - 2

KEY_COLS = ["code_agence", "code_article"]
SUBMISSION_KEY_COLS = ["semaine", "code_agence", "code_article"]


def _sales_history():
    return spark.read.table("features_train").select(
        "semaine",
        "annee",
        "num_semaine",
        "code_agence",
        "code_article",
        F.col("quantite").cast("double").alias("quantite"),
    )


def _pre_validation_sales():
    return _sales_history().filter(F.col("annee") < F.lit(VALIDATION_YEAR))


def _validation_sales():
    return _sales_history().filter(
        (F.col("annee") == F.lit(VALIDATION_YEAR))
        & (F.col("num_semaine").between(VALIDATION_WEEK_MIN, VALIDATION_WEEK_MAX))
    )


def _safe_divide(numerator, denominator):
    return F.when(
        denominator.isNotNull() & (denominator != F.lit(0.0)),
        numerator / denominator,
    ).otherwise(F.lit(None).cast("double"))


def _wape(actual_col="quantite", prediction_col="prediction"):
    absolute_error = F.sum(F.abs(F.col(actual_col) - F.col(prediction_col)))
    actual_volume = F.sum(F.abs(F.col(actual_col)))
    return _safe_divide(absolute_error, actual_volume)


def _metric_aggregations(actual_col="quantite", prediction_col="prediction"):
    absolute_error = F.sum(F.abs(F.col(actual_col) - F.col(prediction_col)))
    signed_error = F.sum(F.col(prediction_col) - F.col(actual_col))
    actual_volume = F.sum(F.abs(F.col(actual_col)))
    predicted_volume = F.sum(F.col(prediction_col))

    return [
        _wape(actual_col, prediction_col).alias("wape"),
        F.count("*").alias("n_rows"),
        F.countDistinct(*KEY_COLS).alias("n_pairs"),
        actual_volume.alias("actual_volume"),
        predicted_volume.alias("predicted_volume"),
        absolute_error.alias("absolute_error"),
        signed_error.alias("signed_error"),
        _safe_divide(signed_error, actual_volume).alias("signed_error_over_actual_volume"),
        F.avg(F.abs(F.col(actual_col) - F.col(prediction_col))).alias("mae"),
        F.avg(F.col(prediction_col) - F.col(actual_col)).alias("mean_bias"),
        F.avg(
            F.when(F.col(prediction_col) > F.col(actual_col), F.lit(1.0)).otherwise(F.lit(0.0))
        ).alias("overprediction_rate"),
        F.sum(
            F.when(F.col(actual_col) == F.lit(0.0), F.abs(F.col(prediction_col))).otherwise(
                F.lit(0.0)
            )
        ).alias("predicted_volume_on_actual_zeros"),
    ]


def _with_actual_volume_segment(df):
    return df.withColumn(
        "actual_volume_segment",
        F.when(F.col("quantite") == F.lit(0.0), F.lit("00_zero"))
        .when(F.col("quantite") <= F.lit(5.0), F.lit("01_low_1_to_5"))
        .when(F.col("quantite") <= F.lit(20.0), F.lit("02_medium_6_to_20"))
        .when(F.col("quantite") <= F.lit(100.0), F.lit("03_high_21_to_100"))
        .otherwise(F.lit("04_extreme_gt_100")),
    )


def _with_validation_calendar_segment(df):
    return df.withColumn(
        "validation_calendar_segment",
        F.when(F.col("num_semaine").between(1, 2), F.lit("00_new_year_restart"))
        .when(F.col("num_semaine").between(14, 23), F.lit("01_spring_build_up"))
        .when(F.col("num_semaine").between(24, 26), F.lit("02_pre_summer_peak"))
        .otherwise(F.lit("03_regular_h1")),
    )


@dp.materialized_view(
    comment=(
        "No-leakage validation predictions for several simple baselines on "
        "2025-W01 to 2025-W26."
    )
)
def validation_predictions():
    train_data = _pre_validation_sales()
    val_data = _validation_sales()

    global_mean = train_data.agg(F.avg("quantite").alias("global_mean"))

    pair_mean = train_data.groupBy(*KEY_COLS).agg(
        F.avg("quantite").alias("pair_mean_all_history")
    )

    pair_median = train_data.groupBy(*KEY_COLS).agg(
        F.expr("percentile_approx(quantite, 0.5, 100)").cast("double").alias(
            "pair_median_all_history"
        )
    )

    recent_pair_mean = (
        train_data.filter(F.col("annee") == F.lit(PREVIOUS_YEAR))
        .groupBy(*KEY_COLS)
        .agg(F.avg("quantite").alias("pair_mean_recent_year"))
    )

    last_year_same_week = (
        train_data.filter(F.col("annee") == F.lit(PREVIOUS_YEAR))
        .groupBy(*(KEY_COLS + ["num_semaine"]))
        .agg(F.avg("quantite").alias("last_year_same_week"))
    )

    seasonal_pair_week_mean = train_data.groupBy(*(KEY_COLS + ["num_semaine"])).agg(
        F.avg("quantite").alias("seasonal_pair_week_mean")
    )

    agency_week_mean = train_data.groupBy("code_agence", "num_semaine").agg(
        F.avg("quantite").alias("agency_week_mean")
    )

    article_week_mean = train_data.groupBy("code_article", "num_semaine").agg(
        F.avg("quantite").alias("article_week_mean")
    )

    pair_trend = (
        train_data.filter(F.col("annee").isin(TREND_REFERENCE_YEAR, PREVIOUS_YEAR))
        .groupBy(*KEY_COLS)
        .agg(
            F.sum(
                F.when(F.col("annee") == F.lit(PREVIOUS_YEAR), F.col("quantite")).otherwise(
                    F.lit(0.0)
                )
            ).alias("volume_previous_year"),
            F.sum(
                F.when(
                    F.col("annee") == F.lit(TREND_REFERENCE_YEAR), F.col("quantite")
                ).otherwise(F.lit(0.0))
            ).alias("volume_reference_year"),
        )
        .withColumn(
            "raw_trend_factor",
            _safe_divide(F.col("volume_previous_year"), F.col("volume_reference_year")),
        )
        .withColumn(
            "trend_factor",
            F.when(F.col("raw_trend_factor").isNull(), F.lit(1.0)).otherwise(
                F.least(F.greatest(F.col("raw_trend_factor"), F.lit(0.25)), F.lit(2.0))
            ),
        )
        .select(*KEY_COLS, "trend_factor")
    )

    scored = (
        val_data.crossJoin(global_mean)
        .join(pair_mean, KEY_COLS, "left")
        .join(pair_median, KEY_COLS, "left")
        .join(recent_pair_mean, KEY_COLS, "left")
        .join(last_year_same_week, KEY_COLS + ["num_semaine"], "left")
        .join(seasonal_pair_week_mean, KEY_COLS + ["num_semaine"], "left")
        .join(agency_week_mean, ["code_agence", "num_semaine"], "left")
        .join(article_week_mean, ["code_article", "num_semaine"], "left")
        .join(pair_trend, KEY_COLS, "left")
        .withColumn("prediction_zero", F.lit(0.0))
        .withColumn("prediction_global_mean", F.coalesce(F.col("global_mean"), F.lit(0.0)))
        .withColumn(
            "prediction_pair_mean_all_history",
            F.coalesce(F.col("pair_mean_all_history"), F.col("prediction_global_mean")),
        )
        .withColumn(
            "prediction_pair_median_all_history",
            F.coalesce(F.col("pair_median_all_history"), F.col("prediction_pair_mean_all_history")),
        )
        .withColumn(
            "prediction_pair_mean_recent_year",
            F.coalesce(F.col("pair_mean_recent_year"), F.col("prediction_pair_mean_all_history")),
        )
        .withColumn(
            "prediction_last_year_same_week",
            F.coalesce(F.col("last_year_same_week"), F.col("prediction_pair_mean_recent_year")),
        )
        .withColumn(
            "prediction_seasonal_pair_week_mean",
            F.coalesce(F.col("seasonal_pair_week_mean"), F.col("prediction_pair_mean_all_history")),
        )
        .withColumn(
            "prediction_agency_week_mean",
            F.coalesce(F.col("agency_week_mean"), F.col("prediction_global_mean")),
        )
        .withColumn(
            "prediction_article_week_mean",
            F.coalesce(F.col("article_week_mean"), F.col("prediction_global_mean")),
        )
        .withColumn(
            "prediction_trend_adjusted_last_year",
            F.coalesce(
                F.col("last_year_same_week") * F.coalesce(F.col("trend_factor"), F.lit(1.0)),
                F.col("prediction_last_year_same_week"),
            ),
        )
        .withColumn(
            "prediction_weighted_seasonal_recent_blend",
            F.lit(0.6) * F.col("prediction_last_year_same_week")
            + F.lit(0.4) * F.col("prediction_pair_mean_recent_year"),
        )
    )

    return (
        scored.select(
            "semaine",
            "annee",
            "num_semaine",
            "code_agence",
            "code_article",
            "quantite",
            F.expr(
                """
                stack(
                    11,
                    'zero_forecast', prediction_zero,
                    'global_mean', prediction_global_mean,
                    'current_dummy_pair_mean', prediction_pair_mean_all_history,
                    'pair_median_all_history', prediction_pair_median_all_history,
                    'pair_mean_recent_year', prediction_pair_mean_recent_year,
                    'last_year_same_week', prediction_last_year_same_week,
                    'seasonal_pair_week_mean', prediction_seasonal_pair_week_mean,
                    'trend_adjusted_last_year', prediction_trend_adjusted_last_year,
                    'agency_week_mean', prediction_agency_week_mean,
                    'article_week_mean', prediction_article_week_mean,
                    'weighted_seasonal_recent_blend', prediction_weighted_seasonal_recent_blend
                ) as (model_name, prediction)
                """
            ),
        )
        .withColumn("prediction", F.greatest(F.col("prediction").cast("double"), F.lit(0.0)))
        .withColumn("absolute_error", F.abs(F.col("quantite") - F.col("prediction")))
        .withColumn("signed_error", F.col("prediction") - F.col("quantite"))
        .withColumn("actual_is_zero", F.col("quantite") == F.lit(0.0))
        .withColumn("prediction_is_zero", F.col("prediction") == F.lit(0.0))
        .withColumn(
            "absolute_error_over_actual",
            _safe_divide(F.col("absolute_error"), F.abs(F.col("quantite"))),
        )
    )


@dp.materialized_view(
    comment="Compatibility WAPE score for the current dummy pair-mean validation model."
)
def wape_score():
    return (
        spark.read.table("validation_predictions")
        .filter(F.col("model_name") == F.lit("current_dummy_pair_mean"))
        .groupBy("model_name")
        .agg(
            _wape().alias("wape"),
            F.count("*").alias("n_validation_samples"),
            F.countDistinct(*KEY_COLS).alias("n_validation_pairs"),
        )
    )


@dp.materialized_view(
    comment="Validation WAPE and bias comparison across leakage-safe baseline candidates."
)
def validation_baseline_comparison():
    return (
        spark.read.table("validation_predictions")
        .groupBy("model_name")
        .agg(*_metric_aggregations())
        .orderBy("wape")
    )


@dp.materialized_view(comment="Validation diagnostics by week to expose seasonal weak spots.")
def validation_wape_by_week():
    return (
        _with_validation_calendar_segment(spark.read.table("validation_predictions"))
        .groupBy("model_name", "annee", "num_semaine", "validation_calendar_segment")
        .agg(*_metric_aggregations())
        .orderBy("model_name", "num_semaine")
    )


@dp.materialized_view(comment="Validation diagnostics by calendar segment.")
def validation_wape_by_calendar_segment():
    return (
        _with_validation_calendar_segment(spark.read.table("validation_predictions"))
        .groupBy("model_name", "validation_calendar_segment")
        .agg(*_metric_aggregations())
        .orderBy("model_name", "validation_calendar_segment")
    )


@dp.materialized_view(comment="Validation diagnostics by agency.")
def validation_wape_by_agency():
    return (
        spark.read.table("validation_predictions")
        .groupBy("model_name", "code_agence")
        .agg(*_metric_aggregations())
        .orderBy("model_name", "wape")
    )


@dp.materialized_view(
    comment=(
        "Validation diagnostics by actual demand bucket, including zero-heavy and "
        "heavy-tail rows."
    )
)
def validation_wape_by_actual_volume_segment():
    return (
        _with_actual_volume_segment(spark.read.table("validation_predictions"))
        .groupBy("model_name", "actual_volume_segment")
        .agg(*_metric_aggregations())
        .orderBy("model_name", "actual_volume_segment")
    )


@dp.materialized_view(
    comment="Validation diagnostics by pair intermittency measured before the validation period."
)
def validation_wape_by_intermittency_segment():
    train_profile = (
        _pre_validation_sales()
        .groupBy(*KEY_COLS)
        .agg(
            F.count("*").alias("train_weeks_observed"),
            F.sum(F.when(F.col("quantite") > F.lit(0.0), F.lit(1)).otherwise(F.lit(0))).alias(
                "train_nonzero_weeks"
            ),
            F.avg(F.when(F.col("quantite") == F.lit(0.0), F.lit(1.0)).otherwise(F.lit(0.0))).alias(
                "train_zero_rate"
            ),
            F.avg("quantite").alias("train_mean_quantity"),
            F.sum("quantite").alias("train_total_quantity"),
        )
        .withColumn(
            "intermittency_segment",
            F.when(F.col("train_zero_rate") >= F.lit(0.95), F.lit("00_very_sparse_ge_95pct_zero"))
            .when(F.col("train_zero_rate") >= F.lit(0.75), F.lit("01_sparse_75_95pct_zero"))
            .when(F.col("train_zero_rate") >= F.lit(0.50), F.lit("02_mixed_50_75pct_zero"))
            .otherwise(F.lit("03_regular_lt_50pct_zero")),
        )
    )

    return (
        spark.read.table("validation_predictions")
        .join(train_profile, KEY_COLS, "left")
        .groupBy("model_name", "intermittency_segment")
        .agg(
            *_metric_aggregations(),
            F.avg("train_zero_rate").alias("avg_train_zero_rate"),
            F.avg("train_nonzero_weeks").alias("avg_train_nonzero_weeks"),
            F.avg("train_total_quantity").alias("avg_train_total_quantity"),
        )
        .orderBy("model_name", "intermittency_segment")
    )


@dp.materialized_view(
    comment="Worst validation errors per candidate, useful for manual inspection."
)
def validation_largest_errors():
    ranked_errors = (
        spark.read.table("validation_predictions")
        .withColumn(
            "error_rank",
            F.row_number().over(Window.partitionBy("model_name").orderBy(F.desc("absolute_error"))),
        )
        .withColumn(
            "signed_error_over_actual",
            _safe_divide(F.col("signed_error"), F.abs(F.col("quantite"))),
        )
    )

    return (
        ranked_errors.filter(F.col("error_rank") <= F.lit(30))
        .select(
            "model_name",
            "error_rank",
            "semaine",
            "code_agence",
            "code_article",
            "quantite",
            "prediction",
            "absolute_error",
            "signed_error",
            "signed_error_over_actual",
            "actual_is_zero",
        )
        .orderBy("model_name", "error_rank")
    )


@dp.materialized_view(
    comment=(
        "Validation split integrity checks: strict temporal split, non-empty "
        "validation, and pair history coverage."
    )
)
def validation_split_checks():
    train_pairs = _pre_validation_sales().select(*KEY_COLS).distinct().withColumn(
        "_seen_in_training", F.lit(1)
    )

    validation_with_history_flag = _validation_sales().join(train_pairs, KEY_COLS, "left")

    summary = validation_with_history_flag.agg(
        F.count("*").alias("n_validation_rows"),
        F.countDistinct(*KEY_COLS).alias("n_validation_pairs"),
        F.sum(F.abs(F.col("quantite"))).alias("validation_actual_volume"),
        F.sum(
            F.when(F.col("_seen_in_training").isNull(), F.lit(1)).otherwise(F.lit(0))
        ).alias("n_validation_rows_without_pair_history"),
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
                F.struct(
                    F.lit("validation_pairs_have_training_history").alias("check_name"),
                    F.when(
                        F.col("n_validation_rows_without_pair_history") == F.lit(0),
                        F.lit("PASS"),
                    )
                    .otherwise(F.lit("WARN"))
                    .alias("status"),
                    F.concat(
                        F.lit("rows_without_pair_history="),
                        F.col("n_validation_rows_without_pair_history").cast("string"),
                        F.lit(", n_validation_pairs="),
                        F.col("n_validation_pairs").cast("string"),
                    ).alias("details"),
                ),
            )
        ).alias("check")
    ).select("check.*")


@dp.materialized_view(
    comment="Submission table sanity checks for the hidden-test prediction output."
)
def submission_prediction_checks():
    test_keys = spark.read.table("features_test").select(*SUBMISSION_KEY_COLS)

    predictions = spark.read.table("predictions").select(
        *SUBMISSION_KEY_COLS,
        F.col("predicted_quantite").cast("double").alias("predicted_quantite"),
    )

    test_summary = test_keys.agg(
        F.count("*").alias("n_test_rows"),
        F.countDistinct(*SUBMISSION_KEY_COLS).alias("n_distinct_test_keys"),
    ).withColumn("n_duplicate_test_rows", F.col("n_test_rows") - F.col("n_distinct_test_keys"))

    prediction_null_key = (
        F.col("semaine").isNull()
        | F.col("code_agence").isNull()
        | F.col("code_article").isNull()
    )

    prediction_summary = predictions.agg(
        F.count("*").alias("n_prediction_rows"),
        F.countDistinct(*SUBMISSION_KEY_COLS).alias("n_distinct_prediction_keys"),
        F.sum(F.when(prediction_null_key, F.lit(1)).otherwise(F.lit(0))).alias(
            "n_prediction_rows_with_null_key"
        ),
        F.sum(
            F.when(F.col("predicted_quantite").isNull(), F.lit(1)).otherwise(F.lit(0))
        ).alias("n_null_predictions"),
        F.sum(
            F.when(
                F.col("predicted_quantite") < F.lit(0.0),
                F.lit(1),
            ).otherwise(F.lit(0))
        ).alias("n_negative_predictions"),
    ).withColumn(
        "n_duplicate_prediction_rows",
        F.col("n_prediction_rows") - F.col("n_distinct_prediction_keys"),
    )

    joined_distinct_keys = (
        test_keys.distinct()
        .withColumn("_in_test", F.lit(1))
        .join(
            predictions.select(*SUBMISSION_KEY_COLS).distinct().withColumn(
                "_in_predictions", F.lit(1)
            ),
            SUBMISSION_KEY_COLS,
            "full",
        )
    )

    coverage_summary = joined_distinct_keys.agg(
        F.sum(
            F.when(
                F.col("_in_test").isNotNull() & F.col("_in_predictions").isNull(), F.lit(1)
            ).otherwise(F.lit(0))
        ).alias("n_missing_prediction_keys"),
        F.sum(
            F.when(
                F.col("_in_test").isNull() & F.col("_in_predictions").isNotNull(), F.lit(1)
            ).otherwise(F.lit(0))
        ).alias("n_extra_prediction_keys"),
    )

    summary = test_summary.crossJoin(prediction_summary).crossJoin(coverage_summary)

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
                    F.lit("unique_test_keys").alias("check_name"),
                    F.when(F.col("n_duplicate_test_rows") == F.lit(0), F.lit("PASS"))
                    .otherwise(F.lit("WARN"))
                    .alias("status"),
                    F.concat(
                        F.lit("duplicate_test_rows="),
                        F.col("n_duplicate_test_rows").cast("string"),
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
                    F.lit("no_null_submission_keys").alias("check_name"),
                    F.when(
                        F.col("n_prediction_rows_with_null_key") == F.lit(0), F.lit("PASS")
                    )
                    .otherwise(F.lit("FAIL"))
                    .alias("status"),
                    F.concat(
                        F.lit("prediction_rows_with_null_key="),
                        F.col("n_prediction_rows_with_null_key").cast("string"),
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


@dp.materialized_view(
    comment=(
        "Hidden-test prediction profile by week, compared with 2024 same-week "
        "training volume."
    )
)
def test_prediction_profile_by_week():
    test_rows = spark.read.table("features_test").select(
        *SUBMISSION_KEY_COLS, "annee", "num_semaine"
    )

    predictions = spark.read.table("predictions").select(
        *SUBMISSION_KEY_COLS, F.col("predicted_quantite").cast("double").alias("predicted_quantite")
    )

    predicted_by_week = (
        test_rows.join(predictions, SUBMISSION_KEY_COLS, "left")
        .groupBy("annee", "num_semaine")
        .agg(
            F.count("*").alias("n_prediction_rows"),
            F.sum("predicted_quantite").alias("predicted_volume"),
            F.avg("predicted_quantite").alias("avg_prediction"),
            F.expr("percentile_approx(predicted_quantite, 0.5, 100)").alias(
                "median_prediction"
            ),
            F.sum(
                F.when(F.col("predicted_quantite") == F.lit(0.0), F.lit(1)).otherwise(F.lit(0))
            ).alias("n_zero_predictions"),
            F.avg(
                F.when(F.col("predicted_quantite") == F.lit(0.0), F.lit(1.0)).otherwise(
                    F.lit(0.0)
                )
            ).alias("prediction_zero_rate"),
        )
    )

    last_year_by_week = (
        _sales_history()
        .filter(F.col("annee") == F.lit(PREVIOUS_YEAR))
        .groupBy("num_semaine")
        .agg(
            F.sum("quantite").alias("actual_volume_same_week_2024"),
            F.avg("quantite").alias("avg_quantity_same_week_2024"),
            F.expr("percentile_approx(quantite, 0.5, 100)").alias(
                "median_quantity_same_week_2024"
            ),
        )
    )

    return (
        predicted_by_week.join(last_year_by_week, "num_semaine", "left")
        .withColumn(
            "predicted_to_2024_volume_ratio",
            _safe_divide(F.col("predicted_volume"), F.col("actual_volume_same_week_2024")),
        )
        .orderBy("annee", "num_semaine")
    )
