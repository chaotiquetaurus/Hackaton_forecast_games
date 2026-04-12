# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Iterative inference
# MAGIC
# MAGIC Loads the registered two-stage LightGBM models, rebuilds target-derived
# MAGIC features week by week, feeds predictions into future lags, and writes the
# MAGIC final leaderboard table.

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

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

from src.iterative import build_iterative_state, score_iterative_horizon

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load training parameters and registered models

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT],
    filter_string="tags.mlflow.runName = 'train_pipeline'",
    order_by=["start_time DESC"],
    max_results=1,
)
if len(runs) == 0:
    raise RuntimeError("No train_pipeline run found; run 03_train_model.py first.")

run_row = runs.iloc[0]

def _run_param(name: str, default=None):
    return run_row.get(f"params.{name}", default)

best_threshold = float(_run_param("best_zero_threshold", 0.55))
best_blend_alpha = float(_run_param("best_blend_alpha", 0.0))
best_zero_iter = _run_param("best_iteration_zero", None)
best_qty_iter = _run_param("best_iteration_qty", None)
best_zero_iter = int(best_zero_iter) if best_zero_iter not in [None, ""] else None
best_qty_iter = int(best_qty_iter) if best_qty_iter not in [None, ""] else None

print(
    f"Using threshold={best_threshold:.2f}, blend_alpha={best_blend_alpha:.2f}, "
    f"zero_iter={best_zero_iter}, qty_iter={best_qty_iter}"
)

client = mlflow.tracking.MlflowClient()

def _latest_version(name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No version found for model {name}")
    return str(max(int(v.version) for v in versions))

v_zero = _latest_version(MLFLOW_MODEL_NAME_ZERO)
v_qty = _latest_version(MLFLOW_MODEL_NAME_QTY)

model_zero = mlflow.lightgbm.load_model(f"models:/{MLFLOW_MODEL_NAME_ZERO}/{v_zero}")
model_qty = mlflow.lightgbm.load_model(f"models:/{MLFLOW_MODEL_NAME_QTY}/{v_qty}")

print(f"Loaded zero_classifier v{v_zero}, qty_regressor v{v_qty}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load known history and final horizon

# COMMAND ----------

META_COLS = ["semaine", "week_id", "annee", "num_sem", "code_agence", "code_article", "quantite", "is_dead_pair"]
feature_cols = list(dict.fromkeys(META_COLS + FEATURES))

history_pd = (
    spark.table(TBL_GOLD_FEATURES)
    .filter(F.col("week_id") <= F.lit(INTERNAL_TEST_END_WEEK_ID))
    .select(*META_COLS)
    .toPandas()
)
final_pd = spark.table(TBL_GOLD_FINAL_INFERENCE).select(*feature_cols).toPandas()

print(f"Known labelled history rows: {len(history_pd):,}")
print(f"Final inference rows: {len(final_pd):,}")

base_state_final = build_iterative_state(history_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Score recursively

# COMMAND ----------

final_iter = score_iterative_horizon(
    base_state=base_state_final,
    horizon_df=final_pd,
    zero_model=model_zero,
    qty_model=model_qty,
    features=FEATURES,
    categorical_features=FEATURES_CATEGORICAL,
    threshold=best_threshold,
    label="final_inference",
    zero_iter=best_zero_iter,
    qty_iter=best_qty_iter,
    blend_alpha=best_blend_alpha,
    feed_rounded=ITERATIVE_FEED_ROUNDED,
)

submission = final_iter[["semaine", "code_agence", "code_article", "prediction_int"]].copy()
submission = submission.rename(columns={"prediction_int": "quantite"})
submission["code_agence"] = submission["code_agence"].astype("int64")
submission["code_article"] = submission["code_article"].astype("int64")
submission["quantite"] = np.clip(submission["quantite"].round(), 0, None).astype("int64")

print(f"Predictions > 0: {(submission['quantite'] > 0).sum():,} / {len(submission):,}")
print(f"Mean predicted: {submission['quantite'].mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write final predictions

# COMMAND ----------

out_sdf = (
    spark.createDataFrame(submission)
    .withColumn("code_agence", F.col("code_agence").cast(LongType()))
    .withColumn("code_article", F.col("code_article").cast(LongType()))
    .withColumn("quantite", F.col("quantite").cast(LongType()))
)

for table_name in [TBL_PREDICTIONS_FINAL, TABLE_PREDICTIONS]:
    (
        out_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    print(f"Wrote {len(submission):,} rows to {table_name}")

print(f"Submission table ready: {TABLE_PREDICTIONS}")

# COMMAND ----------

with mlflow.start_run(run_name="inference"):
    mlflow.log_param("zero_threshold", best_threshold)
    mlflow.log_param("blend_alpha", best_blend_alpha)
    mlflow.log_param("zero_clf_version", v_zero)
    mlflow.log_param("qty_reg_version", v_qty)
    mlflow.log_metric("n_test_rows", len(submission))
    mlflow.log_metric("n_positive_preds", int((submission["quantite"] > 0).sum()))
    mlflow.log_metric("mean_pred", float(submission["quantite"].mean()))
