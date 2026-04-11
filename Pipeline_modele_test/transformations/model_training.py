from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql import types as T


# ---------------------------------------------------------------------------
# LightGBM hurdle model
# ---------------------------------------------------------------------------
#
# Architecture:
# - one LightGBM classifier estimates P(sale > 0), directly addressing the 67%
#   zero rate;
# - one LightGBM Tweedie regressor learns non-negative demand on all rows;
# - one LightGBM positive-demand regressor learns log1p(quantity) on positive
#   rows only;
# - the final forecast is a fixed, conservative blend of the direct Tweedie
#   forecast and the hurdle forecast.
#
# Anti-leakage / anti-overfit practices:
# - all short lags 1..25 are excluded from the feature table;
# - validation is strictly 2025-W01..W26, trained on rows before 2025;
# - metadata joins follow the documented keys;
# - billing features use previous-year aggregates only;
# - LightGBM uses early stopping, subsampling, column sampling, regularization,
#   max_depth, and high min_child_samples.

VALIDATION_YEAR = 2025
VALIDATION_WEEK_MIN = 1
VALIDATION_WEEK_MAX = 26
MIN_TRAIN_YEAR = 2022
MODEL_VERSION = "lightgbm_hurdle_tweedie_v1"

ID_COLS = ["semaine", "code_agence", "code_article"]
TARGET_COL = "quantite"

NUMERIC_FEATURES = [
    "annee",
    "num_semaine",
    "target_month",
    "week_index",
    "week_sin",
    "week_cos",
    "latitude",
    "longitude",
    "poids_en_kg",
    "lag_26",
    "lag_39",
    "lag_52",
    "lag_78",
    "lag_104",
    "rolling_mean_26_52",
    "rolling_sum_26_52",
    "rolling_max_26_52",
    "rolling_std_26_52",
    "zero_rate_26_52",
    "rolling_mean_52_104",
    "rolling_max_52_104",
    "zero_rate_52_104",
    "pair_mean_to_lag_26",
    "pair_nonzero_weeks_to_lag_26",
    "pair_zero_rate_to_lag_26",
    "lag_52_over_lag_104",
    "lag_26_over_lag_52",
    "recent_over_long_mean",
    "fact_prev_year_month_nb_achats",
    "fact_prev_year_month_sum_quantite",
    "fact_prev_year_month_sum_montant",
    "fact_prev_year_month_max_quantite",
    "fact_prev_year_month_avg_quantite_par_achat",
    "fact_prev_year_month_avg_montant_par_achat",
    "fact_prev_year_month_avg_prix_unitaire",
    "fact_prev_year_month_part_pro",
    "fact_prev_year_month_part_direct",
    "fact_prev_year_month_nb_chantiers",
    "fact_prev_year_total_nb_achats",
    "fact_prev_year_total_sum_quantite",
    "fact_prev_year_total_sum_montant",
    "fact_prev_year_total_max_quantite",
    "fact_prev_year_total_nb_chantiers",
    "fact_prev_year_total_part_pro",
    "fact_prev_year_total_part_direct",
    "fact_prev_year_total_avg_prix_unitaire",
]

CATEGORICAL_FEATURES = [
    "code_agence",
    "code_article",
    "region",
    "secteur",
    "metier",
    "code_postal",
    "ville",
    "departement",
    "sous_famille",
    "famille",
    "marque",
    "specialite",
    "unite_vente",
    "article_mdd",
    "gamme",
    "code_fournisseur",
    "has_article_reference",
    "has_prev_year_month_billing",
    "has_prev_year_total_billing",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

VALIDATION_SCHEMA = T.StructType(
    [
        T.StructField("semaine", T.StringType(), False),
        T.StructField("code_agence", T.LongType(), False),
        T.StructField("code_article", T.LongType(), False),
        T.StructField("quantite", T.DoubleType(), False),
        T.StructField("prediction", T.DoubleType(), False),
        T.StructField("direct_prediction", T.DoubleType(), False),
        T.StructField("sale_probability", T.DoubleType(), False),
        T.StructField("conditional_positive_prediction", T.DoubleType(), False),
        T.StructField("hurdle_prediction", T.DoubleType(), False),
        T.StructField("model_version", T.StringType(), False),
    ]
)

PREDICTION_SCHEMA = T.StructType(
    [
        T.StructField("semaine", T.StringType(), False),
        T.StructField("code_agence", T.LongType(), False),
        T.StructField("code_article", T.LongType(), False),
        T.StructField("predicted_quantite", T.DoubleType(), False),
    ]
)


def _load_lightgbm():
    try:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "The Databricks pipeline environment must include the `lightgbm` package "
            "for the production model. Add lightgbm to the pipeline environment "
            "or cluster libraries, then rerun the pipeline."
        ) from exc

    return lgb, np, pd


def _feature_frame():
    return spark.read.table("lightgbm_features")


def _training_rows_for_validation():
    return (
        _feature_frame()
        .filter((F.col("is_test") == F.lit(False)) & (F.col("annee") < F.lit(VALIDATION_YEAR)))
        .filter(F.col("annee") >= F.lit(MIN_TRAIN_YEAR))
    )


def _validation_rows():
    return (
        _feature_frame()
        .filter(
            (F.col("is_test") == F.lit(False))
            & (F.col("annee") == F.lit(VALIDATION_YEAR))
            & (F.col("num_semaine").between(VALIDATION_WEEK_MIN, VALIDATION_WEEK_MAX))
        )
    )


def _training_rows_for_final_model():
    return (
        _feature_frame()
        .filter((F.col("is_test") == F.lit(False)) & (F.col("annee") >= F.lit(MIN_TRAIN_YEAR)))
    )


def _test_rows():
    return _feature_frame().filter(F.col("is_test") == F.lit(True))


def _select_model_columns(df, include_target):
    columns = list(dict.fromkeys(list(ID_COLS) + FEATURE_COLUMNS))
    if include_target:
        columns.append(TARGET_COL)
    return df.select(*[F.col(c) for c in columns])


def _to_pandas(df, include_target):
    return _select_model_columns(df, include_target).toPandas()


def _prepare_pandas_frame(pdf, pd):
    pdf = pdf.copy()

    for col in NUMERIC_FEATURES:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        pdf[col] = pdf[col].astype("object").where(pdf[col].notna(), "__missing__")
        pdf[col] = pdf[col].astype("category")

    pdf["code_agence"] = pdf["code_agence"].astype("category")
    pdf["code_article"] = pdf["code_article"].astype("category")
    return pdf


def _categorical_categories(pdf):
    return {col: pdf[col].cat.categories for col in CATEGORICAL_FEATURES}


def _align_categories(pdf, categories):
    pdf = pdf.copy()
    for col, known_categories in categories.items():
        pdf[col] = pdf[col].cat.set_categories(known_categories)
    return pdf


def _training_weights(y, np):
    positive_mean = max(float(y[y > 0].mean()) if (y > 0).any() else 1.0, 1.0)
    return np.clip(1.0 + np.log1p(y) / np.log1p(positive_mean), 1.0, 8.0)


def _callbacks(lgb, use_early_stopping):
    if use_early_stopping:
        return [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
    return [lgb.log_evaluation(0)]


def _fit_models(train_pdf, val_pdf=None):
    lgb, np, pd = _load_lightgbm()
    train_pdf = _prepare_pandas_frame(train_pdf, pd)
    val_pdf = _prepare_pandas_frame(val_pdf, pd) if val_pdf is not None else None
    categories = _categorical_categories(train_pdf)
    val_pdf = _align_categories(val_pdf, categories) if val_pdf is not None else None

    x_train = train_pdf[FEATURE_COLUMNS]
    y_train = train_pdf[TARGET_COL].astype(float).clip(lower=0.0)
    y_occurrence = (y_train > 0.0).astype(int)
    sample_weight = _training_weights(y_train, np)

    eval_reg = None
    eval_cls = None
    use_early_stopping = val_pdf is not None
    if val_pdf is not None:
        x_val = val_pdf[FEATURE_COLUMNS]
        y_val = val_pdf[TARGET_COL].astype(float).clip(lower=0.0)
        eval_reg = [(x_val, y_val)]
        eval_cls = [(x_val, (y_val > 0.0).astype(int))]

    direct_model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.35,
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=8,
        min_child_samples=250,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    direct_model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=eval_reg,
        eval_metric="l1",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=_callbacks(lgb, use_early_stopping),
    )

    n_positive = int(y_occurrence.sum())
    n_negative = int(len(y_occurrence) - n_positive)
    scale_pos_weight = max(n_negative / max(n_positive, 1), 1.0)
    occurrence_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.04,
        num_leaves=48,
        max_depth=7,
        min_child_samples=350,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        random_state=43,
        n_jobs=-1,
        verbosity=-1,
    )
    occurrence_model.fit(
        x_train,
        y_occurrence,
        eval_set=eval_cls,
        eval_metric="binary_logloss",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=_callbacks(lgb, use_early_stopping),
    )

    positive_mask = y_train > 0.0
    positive_model = None
    if int(positive_mask.sum()) >= 100:
        x_positive = x_train.loc[positive_mask]
        y_positive_log = np.log1p(y_train.loc[positive_mask])
        eval_positive = None
        if val_pdf is not None:
            y_val = val_pdf[TARGET_COL].astype(float).clip(lower=0.0)
            val_positive_mask = y_val > 0.0
            if int(val_positive_mask.sum()) >= 20:
                eval_positive = [(x_val.loc[val_positive_mask], np.log1p(y_val.loc[val_positive_mask]))]

        positive_model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=48,
            max_depth=7,
            min_child_samples=120,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.5,
            random_state=44,
            n_jobs=-1,
            verbosity=-1,
        )
        positive_model.fit(
            x_positive,
            y_positive_log,
            eval_set=eval_positive,
            eval_metric="l1",
            categorical_feature=CATEGORICAL_FEATURES,
            callbacks=_callbacks(lgb, eval_positive is not None),
        )

    return {
        "lgb": lgb,
        "np": np,
        "pd": pd,
        "direct_model": direct_model,
        "occurrence_model": occurrence_model,
        "positive_model": positive_model,
        "categorical_categories": categories,
    }


def _predict_components(models, score_pdf):
    np = models["np"]
    pd = models["pd"]
    score_pdf = _prepare_pandas_frame(score_pdf, pd)
    score_pdf = _align_categories(score_pdf, models["categorical_categories"])
    x_score = score_pdf[FEATURE_COLUMNS]

    direct_prediction = np.clip(models["direct_model"].predict(x_score), 0.0, None)
    sale_probability = np.clip(models["occurrence_model"].predict_proba(x_score)[:, 1], 0.0, 1.0)

    if models["positive_model"] is None:
        conditional_positive_prediction = direct_prediction
    else:
        conditional_positive_prediction = np.clip(
            np.expm1(models["positive_model"].predict(x_score)),
            0.0,
            None,
        )

    hurdle_prediction = sale_probability * conditional_positive_prediction
    raw_prediction = 0.65 * direct_prediction + 0.35 * hurdle_prediction
    sparse_guard = np.where(sale_probability < 0.08, 0.25, np.where(sale_probability < 0.15, 0.65, 1.0))
    prediction = np.clip(raw_prediction * sparse_guard, 0.0, None)

    return {
        "prediction": prediction,
        "direct_prediction": direct_prediction,
        "sale_probability": sale_probability,
        "conditional_positive_prediction": conditional_positive_prediction,
        "hurdle_prediction": hurdle_prediction,
    }


@dp.materialized_view(
    comment="Validation predictions of the single LightGBM hurdle model on 2025-W01..W26"
)
def validation_predictions():
    train_pdf = _to_pandas(_training_rows_for_validation(), include_target=True)
    val_pdf = _to_pandas(_validation_rows(), include_target=True)
    models = _fit_models(train_pdf, val_pdf)
    components = _predict_components(models, val_pdf)

    out_pdf = val_pdf[ID_COLS + [TARGET_COL]].copy()
    out_pdf["prediction"] = components["prediction"]
    out_pdf["direct_prediction"] = components["direct_prediction"]
    out_pdf["sale_probability"] = components["sale_probability"]
    out_pdf["conditional_positive_prediction"] = components["conditional_positive_prediction"]
    out_pdf["hurdle_prediction"] = components["hurdle_prediction"]
    out_pdf["model_version"] = MODEL_VERSION

    return spark.createDataFrame(out_pdf, schema=VALIDATION_SCHEMA)


@dp.materialized_view(
    comment="Final hidden-test predictions from the LightGBM hurdle model"
)
def predictions():
    train_pdf = _to_pandas(_training_rows_for_final_model(), include_target=True)
    test_pdf = _to_pandas(_test_rows(), include_target=False)
    models = _fit_models(train_pdf, None)
    components = _predict_components(models, test_pdf)

    out_pdf = test_pdf[ID_COLS].copy()
    out_pdf["predicted_quantite"] = components["prediction"].round(2)

    return spark.createDataFrame(out_pdf, schema=PREDICTION_SCHEMA)
