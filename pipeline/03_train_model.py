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

# MAGIC %pip install lightgbm==4.3.0
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
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DoubleType, StringType, StructField, StructType

from src.utils import (
    apply_non_iterative_feature_fallbacks,
    wape_numpy,
    wape_lgb_feval,
)

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

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
test_sdf = spark.table(TBL_GOLD_INTERNAL_TEST).select(*cols_needed)

train_pd = train_sdf.toPandas()
val_pd = val_sdf.toPandas()
test_pd = test_sdf.toPandas()

print(f"Train: {len(train_pd):,}   Val: {len(val_pd):,}   Internal test: {len(test_pd):,}")

# Double-check the split has no overlap.
assert train_pd["week_id"].max() < val_pd["week_id"].min(), "Temporal split broken"
assert val_pd["week_id"].max() < test_pd["week_id"].min(), "Temporal split broken"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare X / y / is_zero

# COMMAND ----------

def build_xy(df: pd.DataFrame, use_non_iterative_fallbacks: bool = False):
    X = df[FEATURES].copy()
    if use_non_iterative_fallbacks:
        X = apply_non_iterative_feature_fallbacks(X, FEATURES)
    # Cast categorical encodings to pandas 'category' so LightGBM handles them natively.
    for c in FEATURES_CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df["quantite"].astype(float).values
    is_zero = (y == 0).astype(int)
    return X, y, is_zero

X_tr, y_tr, z_tr = build_xy(train_pd)
X_va, y_va, z_va = build_xy(val_pd, use_non_iterative_fallbacks=True)
X_te, y_te, z_te = build_xy(test_pd, use_non_iterative_fallbacks=True)

print(f"Zero rate train: {z_tr.mean():.3f}   val: {z_va.mean():.3f}   test: {z_te.mean():.3f}")

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
        "internal_test_rows": len(test_pd),
        "train_end": TRAIN_END_WEEK_ID,
        "val_end": VAL_END_WEEK_ID,
        "internal_test_start": INTERNAL_TEST_START_WEEK_ID,
        "internal_test_end": INTERNAL_TEST_END_WEEK_ID,
    })

    # -------- Stage 1 --------
    with mlflow.start_run(run_name="stage1_zero_clf", nested=True) as r1:
        dtrain_z = lgb.Dataset(X_tr, label=z_tr, categorical_feature=FEATURES_CATEGORICAL)
        dval_z = lgb.Dataset(X_va, label=z_va, reference=dtrain_z, categorical_feature=FEATURES_CATEGORICAL)

        model_zero = lgb.train(
            LGB_PARAMS_ZERO,
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
        nz = y_tr > 0
        X_tr_nz = X_tr.loc[nz].reset_index(drop=True)
        y_tr_nz = y_tr[nz]

        y_tr_target = y_tr_nz

        # Validation: also restricted to non-zero rows, but we evaluate the
        # overall pipeline WAPE below on the full val set after combining.
        nz_va = y_va > 0
        X_va_nz = X_va.loc[nz_va].reset_index(drop=True)
        y_va_nz = y_va[nz_va]
        y_va_target = y_va_nz

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

        # regression_l1 = MAE. Unweighted MAE on raw scale aligns with WAPE.
        qty_params = dict(LGB_PARAMS_QTY)
        qty_params["objective"] = "regression_l1"
        qty_params["metric"] = "None"

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

    # -------- Score the labelled internal test set once, after tuning --------
    p_zero_test = model_zero.predict(X_te, num_iteration=model_zero.best_iteration)
    qty_test_raw = model_qty.predict(X_te, num_iteration=model_qty.best_iteration)
    qty_test = np.clip(qty_test_raw, 0.0, None)
    test_final = np.where(p_zero_test > best_threshold, 0.0, qty_test)
    test_wape = wape_numpy(y_te, test_final)
    mlflow.log_metric("internal_test_wape", test_wape)

    test_out = test_pd[["semaine", "code_agence", "code_article", "quantite"]].copy()
    test_out["p_zero"] = p_zero_test
    test_out["qty_pred"] = qty_test
    test_out["prediction"] = test_final

    test_sdf = spark.createDataFrame(test_out, schema=schema)
    (
        test_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TBL_INTERNAL_TEST_PREDICTIONS)
    )
    print(
        f"Wrote internal test predictions to {TBL_INTERNAL_TEST_PREDICTIONS} "
        f"(WAPE={test_wape:.4f})"
    )
