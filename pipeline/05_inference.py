# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Inference
# MAGIC
# MAGIC Applies the ensemble model trained by `03_train_model.py` to the test
# MAGIC period (2025-W27 .. 2025-W52) and displays the final predictions so the
# MAGIC user can download them as CSV from the Databricks UI.
# MAGIC
# MAGIC The training notebook logs every artefact under
# MAGIC `MLFLOW_ENSEMBLE_ARTIFACT_PATH` (LGB clf ensemble, LGB reg ensemble,
# MAGIC XGB reg ensemble, stacker, calibrator, `manifest.json`). This notebook
# MAGIC downloads that folder from the latest `train_pipeline` run, rebuilds the
# MAGIC in-process pipeline, and applies it to the final-inference feature
# MAGIC table. No hyperparameter is hard-coded here — everything is driven by
# MAGIC the manifest.
# MAGIC
# MAGIC **Lag handling at inference:** lags 1..25 are not available when
# MAGIC predicting W27..W52 because they would require target values from
# MAGIC inside the prediction window. We impute them with the pair's expanding
# MAGIC mean via `apply_non_iterative_feature_fallbacks`, same as the
# MAGIC validation/test build.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 xgboost==2.1.3
# MAGIC dbutils.library.restartPython()
# MAGIC # Same NumPy 2.0 compat pins as 03_train_model.py — must match so that
# MAGIC # the saved boosters load cleanly.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import sys
sys.path.append("./")

import json
import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

from src.utils import apply_non_iterative_feature_fallbacks

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download the latest ensemble artifacts

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT],
    filter_string="tags.mlflow.runName = 'train_pipeline'",
    order_by=["start_time DESC"],
    max_results=1,
)
if len(runs) == 0:
    raise RuntimeError("No train_pipeline run found; run 03_train_model.py first.")

run_id = runs.iloc[0]["run_id"]
print(f"Loading ensemble from run {run_id}")

local_dir = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path=MLFLOW_ENSEMBLE_ARTIFACT_PATH,
)
print(f"  → local path: {local_dir}")

with open(os.path.join(local_dir, "manifest.json")) as f:
    manifest = json.load(f)

print(f"  seeds          : {manifest['ensemble_seeds']}")
print(f"  adaptive (b,s) : ({manifest['adaptive_base']}, {manifest['adaptive_slope']})")
print(f"  val WAPE       : {manifest['val_wape']:.4f}")
print(f"  test WAPE      : {manifest['internal_test_wape']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Rebuild the ensemble in-process

# COMMAND ----------

clf_models = [
    lgb.Booster(model_file=os.path.join(local_dir, f"clf_lgb_seed{s}.txt"))
    for s in manifest["ensemble_seeds"]
]
lgb_reg_models = [
    lgb.Booster(model_file=os.path.join(local_dir, f"reg_lgb_seed{s}.txt"))
    for s in manifest["ensemble_seeds"]
]
xgb_reg_models = []
for s in manifest["ensemble_seeds"]:
    b = xgb.Booster()
    b.load_model(os.path.join(local_dir, f"reg_xgb_seed{s}.ubj"))
    xgb_reg_models.append(b)

with open(os.path.join(local_dir, "stacker.pkl"), "rb") as f:
    stacker = pickle.load(f)

calibrator = None
if manifest.get("calibrate_zero_clf"):
    with open(os.path.join(local_dir, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)

print(f"  Loaded {len(clf_models)} clf, {len(lgb_reg_models)} lgb_reg, {len(xgb_reg_models)} xgb_reg")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the final-inference feature rows

# COMMAND ----------

feature_cols = ["semaine", "code_agence", "code_article", "week_id", "is_dead_pair"] + FEATURES
test_df = spark.table(TBL_GOLD_FINAL_INFERENCE).select(*feature_cols).toPandas()
print(f"Final inference rows: {len(test_df):,}   (expected 272 344)")

test_df[FEATURES] = apply_non_iterative_feature_fallbacks(test_df[FEATURES], FEATURES)

X_test = test_df[FEATURES].copy()
for c in FEATURES_CATEGORICAL:
    if c in X_test.columns:
        X_test[c] = X_test[c].astype("category")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score each stage and combine with the stacker

# COMMAND ----------

def _predict_lgb_best(m, X):
    return np.clip(m.predict(X, num_iteration=m.best_iteration), 0.0, None)

def _predict_xgb_best(m, X):
    dm = xgb.DMatrix(X, enable_categorical=True)
    return np.clip(m.predict(dm, iteration_range=(0, m.best_iteration + 1)), 0.0, None)

# Stage 1 — zero probability (ensemble average + optional calibration).
clf_probs = np.mean(
    np.stack([m.predict(X_test, num_iteration=m.best_iteration) for m in clf_models], axis=0),
    axis=0,
)
p_zero = calibrator.transform(clf_probs) if calibrator is not None else clf_probs

# Stage 2a / 2b — LGB and XGB regressor ensembles.
lgb_avg = np.mean(np.stack([_predict_lgb_best(m, X_test) for m in lgb_reg_models], axis=0), axis=0)
xgb_avg = np.mean(np.stack([_predict_xgb_best(m, X_test) for m in xgb_reg_models], axis=0), axis=0)

# Stacker baseline columns are read straight from the feature table.
bl = np.column_stack([
    np.nan_to_num(test_df[c].values.astype(float), nan=0.0)
    for c in manifest["stack_baseline_cols"]
])
blend_raw = np.clip(stacker.predict(np.column_stack([lgb_avg, xgb_avg, bl])), 0.0, None)

# Adaptive threshold per row, then hard dead-pair override.
zero_rate = np.nan_to_num(test_df[manifest["zero_rate_feature"]].values.astype(float), nan=0.0)
clip_lo, clip_hi = manifest["adaptive_clip"]
effective_thr = np.clip(
    manifest["adaptive_base"] - manifest["adaptive_slope"] * zero_rate,
    clip_lo, clip_hi,
)

final = np.where(p_zero > effective_thr, 0.0, blend_raw)
if manifest.get("dead_pair_force_zero", True):
    final = np.where(test_df["is_dead_pair"].values == 1, 0.0, final)

final_int = np.clip(np.round(final), 0, None).astype(np.int64)
print(f"Predictions > 0: {(final_int > 0).sum():,}  /  total: {len(final_int):,}")
print(f"Mean predicted : {final_int.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Write predictions_final and display for CSV download

# COMMAND ----------

predictions_pd = test_df[["semaine", "code_agence", "code_article"]].copy()
predictions_pd["quantite"] = final_int

predictions_sdf = (
    spark.createDataFrame(predictions_pd)
    .withColumn("code_agence",  F.col("code_agence").cast(LongType()))
    .withColumn("code_article", F.col("code_article").cast(LongType()))
    .withColumn("quantite",     F.col("quantite").cast(LongType()))
)

(
    predictions_sdf.write
    .format("delta").mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(TBL_PREDICTIONS_FINAL)
)
print(f"Wrote {TBL_PREDICTIONS_FINAL}")

# Same idea as 6: we can't write to the leaderboard folder directly, so we
# display and let the user download via the UI.
out_sdf = spark.table(TBL_PREDICTIONS_FINAL)
print(f"{out_sdf.count():,} rows")
display(out_sdf)
print("Use the download button above to save as CSV")

# COMMAND ----------

with mlflow.start_run(run_name="inference"):
    mlflow.log_param("train_run_id", run_id)
    mlflow.log_param("ensemble_seeds", json.dumps(manifest["ensemble_seeds"]))
    mlflow.log_metric("n_positive_preds", int((final_int > 0).sum()))
    mlflow.log_metric("mean_pred", float(final_int.mean()))
