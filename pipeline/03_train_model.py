# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Train the two-stage model
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **Stage 1** — a binary LightGBM classifier predicts `P(quantite = 0)`.
# MAGIC - **Stage 2** — a LightGBM regressor (trained only on rows where
# MAGIC   `quantite > 0`) predicts the expected quantity *conditional* on being
# MAGIC   non-zero, using raw `quantite` as the target with MAE loss (regression_l1).
# MAGIC - At scoring time: `pred = 0 if p_zero > threshold else reg_pred`.
# MAGIC - The threshold is tuned on the validation set to minimise WAPE.
# MAGIC
# MAGIC **Inputs:** `workspace.default.feature_table`.
# MAGIC **Outputs:**
# MAGIC - Both models logged to MLflow + registered in the Model Registry.
# MAGIC - The best zero-threshold logged as a run parameter.
# MAGIC - A validation predictions Delta table used by 04_evaluate.py.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install lightgbm==4.3.0 optuna==3.6.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import sys
sys.path.append("./")

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from mlflow.models import infer_signature
from optuna.samplers import TPESampler
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DoubleType, StringType, StructField, StructType

from src.utils import wape_numpy, wape_lgb_feval, prepare_features

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the pre-split feature tables from the Lakeflow pipeline
# MAGIC
# MAGIC The DLT pipeline (`pipeline_lakeflow.py`) produces `gold_train` and
# MAGIC `gold_validation` with the temporal split already applied. We just
# MAGIC project the columns we need and move to pandas for LightGBM.

# COMMAND ----------

cols_needed = (
    ["semaine", "code_agence", "code_article", "week_id", "quantite", "is_dead_pair"]
    + FEATURES
)

train_sdf = spark.table(TBL_GOLD_TRAIN).select(*cols_needed)
val_sdf = spark.table(TBL_GOLD_VAL).select(*cols_needed)

train_pd = train_sdf.toPandas()
val_pd = val_sdf.toPandas()

print(f"Train: {len(train_pd):,}   Val: {len(val_pd):,}")

# Double-check the split has no overlap.
assert train_pd["week_id"].max() < val_pd["week_id"].min(), "Temporal split broken"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1b. Feature preparation (clipping + derived features)
# MAGIC
# MAGIC `prepare_features` clips extreme ratio/CV values to prevent overfit on
# MAGIC pathological splits, caps `pair_last_nonzero_gap` at 104 weeks, and adds
# MAGIC two derived features: `detrended_lag52` (trend-corrected YoY lag) and
# MAGIC `demand_profile` (Syntetos-Boylan 4-class categorisation).

# COMMAND ----------

train_pd = prepare_features(train_pd)
val_pd   = prepare_features(val_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare X / y / is_zero

# COMMAND ----------

def build_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    # Cast categorical encodings to pandas 'category' so LightGBM handles them natively.
    for c in FEATURES_CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df["quantite"].astype(float).values
    is_zero = (y == 0).astype(int)
    return X, y, is_zero

X_tr, y_tr, z_tr = build_xy(train_pd)
X_va, y_va, z_va = build_xy(val_pd)

print(f"Zero rate train: {z_tr.mean():.3f}   val: {z_va.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2b. Optuna helpers

# COMMAND ----------

def _sample_rows(X: pd.DataFrame, *arrays, max_rows=None, seed=SEED):
    """Return a deterministic row sample for tuning; validation stays full-size."""
    if max_rows is None or len(X) <= max_rows:
        return (X, *arrays)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=int(max_rows), replace=False))
    sampled = [X.iloc[idx].reset_index(drop=True)]
    for arr in arrays:
        if isinstance(arr, pd.Series):
            sampled.append(arr.iloc[idx].reset_index(drop=True))
        else:
            sampled.append(np.asarray(arr)[idx])
    return tuple(sampled)


def _log_lgb_params(params: dict, prefix: str):
    mlflow.log_params({f"{prefix}_{k}": v for k, v in sorted(params.items())})


def _log_optuna_study(study: optuna.Study, prefix: str):
    mlflow.log_metric(f"{prefix}_best_value", float(study.best_value))
    mlflow.log_param(f"{prefix}_best_trial", int(study.best_trial.number))
    mlflow.log_params({f"{prefix}_best_{k}": v for k, v in study.best_params.items()})

    trials_path = f"/tmp/{prefix}_optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    mlflow.log_artifact(trials_path)


def _base_qty_params():
    # regression_l1 = MAE. Unweighted MAE on raw scale aligns with WAPE.
    params = dict(LGB_PARAMS_QTY)
    params["objective"] = "regression_l1"
    params["metric"] = "None"
    params.pop("tweedie_variance_power", None)
    return params


def _suggest_zero_params(trial: optuna.Trial):
    params = dict(LGB_PARAMS_ZERO)
    params.update({
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 511, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12, 14, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.60, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.60, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.20),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.30, 3.0, log=True),
        "feature_pre_filter": False,
    })
    return params


def _suggest_qty_params(trial: optuna.Trial):
    params = _base_qty_params()
    params.update({
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 63, 511, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 8, 10, 12, 14, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.60, 0.95),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.60, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.20),
        "feature_pre_filter": False,
    })
    return params


def tune_zero_classifier(X_train, z_train, X_val, z_val):
    X_tune, z_tune = _sample_rows(
        X_train,
        z_train,
        max_rows=OPTUNA_TUNING_MAX_TRAIN_ROWS_ZERO,
        seed=SEED,
    )
    print(f"Optuna zero classifier: tuning on {len(X_tune):,} train rows, {len(X_val):,} val rows")

    dtrain = lgb.Dataset(
        X_tune,
        label=z_tune,
        categorical_feature=FEATURES_CATEGORICAL,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val,
        label=z_val,
        reference=dtrain,
        categorical_feature=FEATURES_CATEGORICAL,
        free_raw_data=False,
    )

    def objective(trial):
        params = _suggest_zero_params(trial)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=OPTUNA_NUM_ROUNDS_ZERO,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=OPTUNA_EARLY_STOP_ZERO,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        p_zero = model.predict(X_val, num_iteration=model.best_iteration)
        val_logloss = float(
            -np.mean(
                z_val * np.log(np.clip(p_zero, 1e-7, 1 - 1e-7))
                + (1 - z_val) * np.log(np.clip(1 - p_zero, 1e-7, 1 - 1e-7))
            )
        )
        trial.set_user_attr("best_iteration", int(model.best_iteration or OPTUNA_NUM_ROUNDS_ZERO))
        return val_logloss

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS_ZERO,
        timeout=OPTUNA_TIMEOUT_SECONDS_ZERO,
        show_progress_bar=False,
    )

    best_params = dict(LGB_PARAMS_ZERO)
    best_params.update(study.best_params)
    best_params["feature_pre_filter"] = False
    _log_optuna_study(study, "zero")
    _log_lgb_params(best_params, "zero_tuned")
    print(f"Best zero logloss: {study.best_value:.5f}  params: {study.best_params}")
    return best_params


def tune_qty_regressor(X_train, y_train, X_val, y_val):
    X_tune, y_tune = _sample_rows(
        X_train,
        y_train,
        max_rows=OPTUNA_TUNING_MAX_TRAIN_ROWS_QTY,
        seed=SEED + 17,
    )
    print(f"Optuna quantity regressor: tuning on {len(X_tune):,} train rows, {len(X_val):,} val rows")

    dtrain = lgb.Dataset(
        X_tune,
        label=y_tune,
        categorical_feature=FEATURES_CATEGORICAL,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=FEATURES_CATEGORICAL,
        free_raw_data=False,
    )

    def objective(trial):
        params = _suggest_qty_params(trial)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=OPTUNA_NUM_ROUNDS_QTY,
            valid_sets=[dval],
            valid_names=["val"],
            feval=wape_lgb_feval,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=OPTUNA_EARLY_STOP_QTY,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        qty_pred = np.clip(model.predict(X_val, num_iteration=model.best_iteration), 0.0, None)
        val_wape = wape_numpy(y_val, qty_pred)
        trial.set_user_attr("best_iteration", int(model.best_iteration or OPTUNA_NUM_ROUNDS_QTY))
        return val_wape

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED + 17))
    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS_QTY,
        timeout=OPTUNA_TIMEOUT_SECONDS_QTY,
        show_progress_bar=False,
    )

    best_params = _base_qty_params()
    best_params.update(study.best_params)
    best_params["feature_pre_filter"] = False
    _log_optuna_study(study, "qty")
    _log_lgb_params(best_params, "qty_tuned")
    print(f"Best quantity WAPE: {study.best_value:.5f}  params: {study.best_params}")
    return best_params

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Stage 1 — zero classifier
# MAGIC
# MAGIC The target is `is_zero`. We log the val logloss and AUC, and later
# MAGIC pull the probabilities to sweep the threshold.

# COMMAND ----------

with mlflow.start_run(run_name="train_pipeline") as parent_run:
    mlflow.log_params({
        "n_features": len(FEATURES),
        "train_rows": len(train_pd),
        "val_rows": len(val_pd),
        "train_end": VAL_START_WEEK_ID,
        "val_end": VAL_END_WEEK_ID,
        "hyperparameter_tuning": RUN_HYPERPARAMETER_TUNING,
        "optuna_n_trials_zero": OPTUNA_N_TRIALS_ZERO if RUN_HYPERPARAMETER_TUNING else 0,
        "optuna_n_trials_qty": OPTUNA_N_TRIALS_QTY if RUN_HYPERPARAMETER_TUNING else 0,
        "optuna_tuning_rows_zero": OPTUNA_TUNING_MAX_TRAIN_ROWS_ZERO or "all",
        "optuna_tuning_rows_qty": OPTUNA_TUNING_MAX_TRAIN_ROWS_QTY or "all",
    })

    # Prepare the non-zero subset once. It is used for quantity tuning and
    # for the final quantity model.
    nz = y_tr > 0
    X_tr_nz = X_tr.loc[nz].reset_index(drop=True)
    y_tr_nz = y_tr[nz]
    y_tr_target = y_tr_nz

    nz_va = y_va > 0
    X_va_nz = X_va.loc[nz_va].reset_index(drop=True)
    y_va_nz = y_va[nz_va]
    y_va_target = y_va_nz

    zero_params = dict(LGB_PARAMS_ZERO)
    qty_params = _base_qty_params()

    if RUN_HYPERPARAMETER_TUNING:
        with mlflow.start_run(run_name="tune_zero_clf", nested=True):
            zero_params = tune_zero_classifier(X_tr, z_tr, X_va, z_va)

        with mlflow.start_run(run_name="tune_qty_regressor", nested=True):
            qty_params = tune_qty_regressor(X_tr_nz, y_tr_target, X_va_nz, y_va_target)
    else:
        print("Optuna tuning disabled; using LGB params from 00_config.py")

    # -------- Stage 1 --------
    with mlflow.start_run(run_name="stage1_zero_clf", nested=True) as r1:
        dtrain_z = lgb.Dataset(X_tr, label=z_tr, categorical_feature=FEATURES_CATEGORICAL)
        dval_z = lgb.Dataset(X_va, label=z_va, reference=dtrain_z, categorical_feature=FEATURES_CATEGORICAL)

        model_zero = lgb.train(
            zero_params,
            dtrain_z,
            num_boost_round=LGB_NUM_ROUNDS_ZERO,
            valid_sets=[dtrain_z, dval_z],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=LGB_EARLY_STOP_ZERO, first_metric_only=True),
                lgb.log_evaluation(period=100),
            ],
        )

        p_zero_val = model_zero.predict(X_va, num_iteration=model_zero.best_iteration)

        _log_lgb_params(zero_params, "zero")
        val_logloss = float(
            -np.mean(
                z_va * np.log(np.clip(p_zero_val, 1e-7, 1 - 1e-7))
                + (1 - z_va) * np.log(np.clip(1 - p_zero_val, 1e-7, 1 - 1e-7))
            )
        )
        mlflow.log_metric("val_logloss", val_logloss)
        mlflow.log_param("best_iteration_zero", model_zero.best_iteration)
        # Cast category cols to int for signature/input_example (MLflow can't serialize category dtype)
        X_tr_clean = X_tr.head(5).copy()
        for c in FEATURES_CATEGORICAL:
            if c in X_tr_clean.columns:
                X_tr_clean[c] = X_tr_clean[c].astype(int)
        sig_zero = infer_signature(X_tr_clean, model_zero.predict(X_tr.head(5)))
        mlflow.lightgbm.log_model(model_zero, artifact_path="zero_classifier", registered_model_name=MLFLOW_MODEL_NAME_ZERO, signature=sig_zero, input_example=X_tr_clean.head(1))

        fi_zero = pd.DataFrame({
            "feature": FEATURES,
            "gain": model_zero.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        fi_zero.to_csv("/tmp/fi_zero.csv", index=False)
        mlflow.log_artifact("/tmp/fi_zero.csv")

    # -------- Stage 2 — MAE on raw quantite (WAPE-aligned) --------
    with mlflow.start_run(run_name="stage2_qty_regressor", nested=True) as r2:
        # Train only on rows where quantite > 0.
        # Target is raw quantite (not log1p) — MAE on raw scale is
        # equivalent to minimising WAPE since the denominator is constant.
        # Validation: also restricted to non-zero rows, but we evaluate the
        # overall pipeline WAPE below on the full val set after combining.
        dtrain_q = lgb.Dataset(
            X_tr_nz,
            label=y_tr_target,
            categorical_feature=FEATURES_CATEGORICAL,
        )
        dval_q = lgb.Dataset(
            X_va_nz,
            label=y_va_target,
            reference=dtrain_q,
            categorical_feature=FEATURES_CATEGORICAL,
        )

        model_qty = lgb.train(
            qty_params,
            dtrain_q,
            num_boost_round=LGB_NUM_ROUNDS_QTY,
            valid_sets=[dtrain_q, dval_q],
            valid_names=["train", "val"],
            feval=wape_lgb_feval,
            callbacks=[
                lgb.early_stopping(stopping_rounds=LGB_EARLY_STOP_QTY, first_metric_only=True),
                lgb.log_evaluation(period=100),
            ],
        )

        _log_lgb_params(qty_params, "qty")
        mlflow.log_param("best_iteration_qty", model_qty.best_iteration)
        # Cast category cols to int for signature/input_example
        X_nz_clean = X_tr_nz.head(5).copy()
        for c in FEATURES_CATEGORICAL:
            if c in X_nz_clean.columns:
                X_nz_clean[c] = X_nz_clean[c].astype(int)
        sig_qty = infer_signature(X_nz_clean, model_qty.predict(X_tr_nz.head(5)))
        mlflow.lightgbm.log_model(model_qty, artifact_path="qty_regressor", registered_model_name=MLFLOW_MODEL_NAME_QTY, signature=sig_qty, input_example=X_nz_clean.head(1))

        fi_qty = pd.DataFrame({
            "feature": FEATURES,
            "gain": model_qty.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        fi_qty.to_csv("/tmp/fi_qty.csv", index=False)
        mlflow.log_artifact("/tmp/fi_qty.csv")

    # -------- Combine on the full val set and sweep threshold --------
    # Predict the quantity on *all* val rows (even those where p_zero is high)
    # — the gate is applied after. Predictions are already in raw scale.
    qty_pred_raw = model_qty.predict(X_va, num_iteration=model_qty.best_iteration)
    qty_pred = np.clip(qty_pred_raw, 0.0, None)

    best_threshold = None
    best_wape = float("inf")
    rows = []
    for thr in ZERO_THRESHOLD_GRID:
        final = np.where(p_zero_val > thr, 0.0, qty_pred)
        wape = wape_numpy(y_va, final)
        rows.append({"threshold": thr, "wape": wape})
        if wape < best_wape:
            best_wape = wape
            best_threshold = thr

    threshold_df = pd.DataFrame(rows)
    print(threshold_df)
    print(f"Best threshold: {best_threshold}  ->  WAPE = {best_wape:.4f}")

    mlflow.log_param("best_zero_threshold", best_threshold)
    mlflow.log_metric("val_wape", best_wape)
    threshold_df.to_csv("/tmp/threshold_sweep.csv", index=False)
    mlflow.log_artifact("/tmp/threshold_sweep.csv")

    # -------- Write validation predictions for 04_evaluate --------
    val_final = np.where(p_zero_val > best_threshold, 0.0, qty_pred)
    val_out = val_pd[["semaine", "code_agence", "code_article", "quantite"]].copy()
    val_out["p_zero"] = p_zero_val
    val_out["qty_pred"] = qty_pred
    val_out["prediction"] = val_final

    schema = StructType([
        StructField("semaine", StringType()),
        StructField("code_agence", LongType()),
        StructField("code_article", LongType()),
        StructField("quantite", LongType()),
        StructField("p_zero", DoubleType()),
        StructField("qty_pred", DoubleType()),
        StructField("prediction", DoubleType()),
    ])
    val_sdf = spark.createDataFrame(val_out, schema=schema)
    (
        val_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TBL_VAL_PREDICTIONS)
    )
    print(f"Wrote validation predictions to {TBL_VAL_PREDICTIONS}")
