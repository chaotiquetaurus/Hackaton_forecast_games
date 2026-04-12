# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Hyperparameter Tuning (Optuna TPE)
# MAGIC
# MAGIC Replaces `03_train_model.py` for tuning runs. Same two-stage hurdle
# MAGIC architecture, but searches for the best LightGBM hyperparameters using
# MAGIC Bayesian optimisation (Optuna TPE sampler).
# MAGIC
# MAGIC **Flow:**
# MAGIC 1. Load gold_train / gold_val (unchanged from 03_train_model).
# MAGIC 2. Run Optuna study (~25 trials, reduced boost rounds for speed).
# MAGIC 3. Retrain the best config at full boost budget.
# MAGIC 4. Log final models + val predictions to MLflow (04_evaluate works as-is).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install optuna lightgbm==4.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import sys
sys.path.append("./")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DoubleType, StringType, StructField, StructType

from src.utils import wape_numpy, wape_lgb_feval, prepare_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data (same as 03_train_model)

# COMMAND ----------

_DERIVED_FEATURES = {"detrended_lag52", "demand_profile"}
cols_needed = (
    ["semaine", "code_agence", "code_article", "week_id", "quantite", "is_dead_pair"]
    + [f for f in FEATURES if f not in _DERIVED_FEATURES]
)

train_sdf = spark.table(TBL_GOLD_TRAIN).select(*cols_needed)
val_sdf = spark.table(TBL_GOLD_VAL).select(*cols_needed)

train_pd = train_sdf.toPandas()
val_pd = val_sdf.toPandas()

train_pd = prepare_features(train_pd)
val_pd = prepare_features(val_pd)

print(f"Train: {len(train_pd):,}   Val: {len(val_pd):,}")
assert train_pd["week_id"].max() < val_pd["week_id"].min(), "Temporal split broken"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare X / y

# COMMAND ----------

def build_xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    for c in FEATURES_CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df["quantite"].astype(float).values
    is_zero = (y == 0).astype(int)
    return X, y, is_zero

X_tr, y_tr, z_tr = build_xy(train_pd)
X_va, y_va, z_va = build_xy(val_pd)

# Pre-filter non-zero rows once (reused by every trial).
nz_tr = y_tr > 0
X_tr_nz = X_tr.loc[nz_tr].reset_index(drop=True)
y_tr_nz = y_tr[nz_tr]

nz_va = y_va > 0
X_va_nz = X_va.loc[nz_va].reset_index(drop=True)
y_va_nz = y_va[nz_va]

print(f"Zero rate train: {z_tr.mean():.3f}   val: {z_va.mean():.3f}")
print(f"Non-zero train: {len(X_tr_nz):,}   val: {len(X_va_nz):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Optuna objective

# COMMAND ----------

# Reduced rounds for search speed — best trial gets full budget later.
SEARCH_ROUNDS_ZERO = 1000
SEARCH_ROUNDS_QTY = 1500
SEARCH_EARLY_STOP = 50


def objective(trial: optuna.Trial) -> float:
    """Train both stages with suggested params, return val WAPE."""

    # --- Stage 1 params ---
    lr_zero = trial.suggest_float("zero_lr", 0.02, 0.12, log=True)
    leaves_zero = trial.suggest_int("zero_num_leaves", 63, 255, step=2)
    min_child_zero = trial.suggest_int("zero_min_child", 20, 100)
    ff_zero = trial.suggest_float("zero_feature_fraction", 0.6, 0.95)

    params_zero = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": lr_zero,
        "num_leaves": leaves_zero,
        "min_child_samples": min_child_zero,
        "feature_fraction": ff_zero,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.5,
        "n_jobs": -1,
        "seed": SEED,
        "verbose": -1,
    }

    # --- Stage 2 params ---
    lr_qty = trial.suggest_float("qty_lr", 0.01, 0.10, log=True)
    leaves_qty = trial.suggest_int("qty_num_leaves", 127, 511, step=2)
    min_child_qty = trial.suggest_int("qty_min_child", 10, 80)
    ff_qty = trial.suggest_float("qty_feature_fraction", 0.6, 0.95)

    params_qty = {
        "objective": "regression_l1",
        "metric": "None",
        "learning_rate": lr_qty,
        "num_leaves": leaves_qty,
        "min_child_samples": min_child_qty,
        "feature_fraction": ff_qty,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 511,
        "n_jobs": -1,
        "seed": SEED,
        "verbose": -1,
    }

    # --- Train Stage 1 ---
    dtrain_z = lgb.Dataset(X_tr, label=z_tr, categorical_feature=FEATURES_CATEGORICAL)
    dval_z = lgb.Dataset(X_va, label=z_va, reference=dtrain_z, categorical_feature=FEATURES_CATEGORICAL)

    model_zero = lgb.train(
        params_zero,
        dtrain_z,
        num_boost_round=SEARCH_ROUNDS_ZERO,
        valid_sets=[dval_z],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=SEARCH_EARLY_STOP, first_metric_only=True),
        ],
    )
    p_zero_val = model_zero.predict(X_va, num_iteration=model_zero.best_iteration)

    # --- Train Stage 2 ---
    dtrain_q = lgb.Dataset(X_tr_nz, label=y_tr_nz, categorical_feature=FEATURES_CATEGORICAL)
    dval_q = lgb.Dataset(X_va_nz, label=y_va_nz, reference=dtrain_q, categorical_feature=FEATURES_CATEGORICAL)

    model_qty = lgb.train(
        params_qty,
        dtrain_q,
        num_boost_round=SEARCH_ROUNDS_QTY,
        valid_sets=[dval_q],
        valid_names=["val"],
        feval=wape_lgb_feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=SEARCH_EARLY_STOP, first_metric_only=True),
        ],
    )
    qty_pred = np.clip(
        model_qty.predict(X_va, num_iteration=model_qty.best_iteration),
        0.0, None,
    )

    # --- Sweep threshold ---
    best_wape = float("inf")
    for thr in ZERO_THRESHOLD_GRID:
        final = np.where(p_zero_val > thr, 0.0, qty_pred)
        w = wape_numpy(y_va, final)
        if w < best_wape:
            best_wape = w
            trial.set_user_attr("best_threshold", thr)

    trial.set_user_attr("best_iter_zero", model_zero.best_iteration)
    trial.set_user_attr("best_iter_qty", model_qty.best_iteration)

    return best_wape

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Run the study

# COMMAND ----------

N_TRIALS = 25

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="sgdb_hp_tuning",
)

print(f"Starting Optuna study — {N_TRIALS} trials")
print("=" * 60)

# Callback to print progress after each trial.
def _print_trial(study, trial):
    print(
        f"  Trial {trial.number:>2d}  WAPE={trial.value:.4f}  "
        f"(best so far: {study.best_value:.4f} @ trial {study.best_trial.number})"
    )

study.optimize(objective, n_trials=N_TRIALS, callbacks=[_print_trial])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Results summary

# COMMAND ----------

best = study.best_trial
print("=" * 60)
print(f"BEST TRIAL: #{best.number}   WAPE = {best.value:.4f}")
print(f"  threshold = {best.user_attrs['best_threshold']}")
print("=" * 60)
print("Best hyperparameters:")
for k, v in best.params.items():
    print(f"  {k:30s} = {v}")

# Show top 5 trials.
trials_df = study.trials_dataframe().sort_values("value")
print("\nTop 5 trials:")
print(trials_df[["number", "value"] + [c for c in trials_df.columns if c.startswith("params_")]].head(5).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Retrain best config at full boost budget & log to MLflow

# COMMAND ----------

bp = best.params

final_params_zero = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": bp["zero_lr"],
    "num_leaves": bp["zero_num_leaves"],
    "min_child_samples": bp["zero_min_child"],
    "feature_fraction": bp["zero_feature_fraction"],
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.5,
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}

final_params_qty = {
    "objective": "regression_l1",
    "metric": "None",
    "learning_rate": bp["qty_lr"],
    "num_leaves": bp["qty_num_leaves"],
    "min_child_samples": bp["qty_min_child"],
    "feature_fraction": bp["qty_feature_fraction"],
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 511,
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}

with mlflow.start_run(run_name="hp_tuning_best") as parent_run:
    mlflow.log_params({
        "n_features": len(FEATURES),
        "train_rows": len(train_pd),
        "val_rows": len(val_pd),
        "train_end": VAL_START_WEEK_ID,
        "val_end": VAL_END_WEEK_ID,
        "optuna_n_trials": N_TRIALS,
        "optuna_best_trial": best.number,
    })
    mlflow.log_params({f"tuned_{k}": v for k, v in bp.items()})

    # -------- Stage 1 (full budget) --------
    with mlflow.start_run(run_name="stage1_zero_clf", nested=True):
        dtrain_z = lgb.Dataset(X_tr, label=z_tr, categorical_feature=FEATURES_CATEGORICAL)
        dval_z = lgb.Dataset(X_va, label=z_va, reference=dtrain_z, categorical_feature=FEATURES_CATEGORICAL)

        model_zero = lgb.train(
            final_params_zero,
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
        mlflow.log_param("best_iteration_zero", model_zero.best_iteration)

        X_tr_clean = X_tr.head(5).copy()
        for c in FEATURES_CATEGORICAL:
            if c in X_tr_clean.columns:
                X_tr_clean[c] = X_tr_clean[c].astype(int)
        sig_zero = infer_signature(X_tr_clean, model_zero.predict(X_tr.head(5)))
        mlflow.lightgbm.log_model(
            model_zero, artifact_path="zero_classifier",
            registered_model_name=MLFLOW_MODEL_NAME_ZERO,
            signature=sig_zero, input_example=X_tr_clean.head(1),
        )

        fi_zero = pd.DataFrame({
            "feature": FEATURES,
            "gain": model_zero.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        fi_zero.to_csv("/tmp/fi_zero.csv", index=False)
        mlflow.log_artifact("/tmp/fi_zero.csv")

    # -------- Stage 2 (full budget) --------
    with mlflow.start_run(run_name="stage2_qty_regressor", nested=True):
        dtrain_q = lgb.Dataset(X_tr_nz, label=y_tr_nz, categorical_feature=FEATURES_CATEGORICAL)
        dval_q = lgb.Dataset(X_va_nz, label=y_va_nz, reference=dtrain_q, categorical_feature=FEATURES_CATEGORICAL)

        model_qty = lgb.train(
            final_params_qty,
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
        mlflow.log_param("best_iteration_qty", model_qty.best_iteration)

        X_nz_clean = X_tr_nz.head(5).copy()
        for c in FEATURES_CATEGORICAL:
            if c in X_nz_clean.columns:
                X_nz_clean[c] = X_nz_clean[c].astype(int)
        sig_qty = infer_signature(X_nz_clean, model_qty.predict(X_tr_nz.head(5)))
        mlflow.lightgbm.log_model(
            model_qty, artifact_path="qty_regressor",
            registered_model_name=MLFLOW_MODEL_NAME_QTY,
            signature=sig_qty, input_example=X_nz_clean.head(1),
        )

        fi_qty = pd.DataFrame({
            "feature": FEATURES,
            "gain": model_qty.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        fi_qty.to_csv("/tmp/fi_qty.csv", index=False)
        mlflow.log_artifact("/tmp/fi_qty.csv")

    # -------- Threshold sweep on final models --------
    qty_pred = np.clip(
        model_qty.predict(X_va, num_iteration=model_qty.best_iteration),
        0.0, None,
    )

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
    val_sdf_out = spark.createDataFrame(val_out, schema=schema)
    (
        val_sdf_out.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TBL_VAL_PREDICTIONS)
    )
    print(f"Wrote validation predictions to {TBL_VAL_PREDICTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Recommended config update
# MAGIC
# MAGIC Copy-paste these into `00_config.py` once you're happy with the results.

# COMMAND ----------

print("=" * 60)
print("  PASTE INTO 00_config.py")
print("=" * 60)
print(f"""
LGB_PARAMS_ZERO = {{
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": {bp['zero_lr']:.4f},
    "num_leaves": {bp['zero_num_leaves']},
    "min_child_samples": {bp['zero_min_child']},
    "feature_fraction": {bp['zero_feature_fraction']:.3f},
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.5,
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}}

LGB_PARAMS_QTY = {{
    "objective": "regression_l1",
    "metric": "None",
    "learning_rate": {bp['qty_lr']:.4f},
    "num_leaves": {bp['qty_num_leaves']},
    "min_child_samples": {bp['qty_min_child']},
    "feature_fraction": {bp['qty_feature_fraction']:.3f},
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 511,
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}}
""")
print(f"Best WAPE: {best_wape:.4f}  (threshold: {best_threshold})")
