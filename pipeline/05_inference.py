# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Inference
# MAGIC
# MAGIC Loads the latest registered two-stage LightGBM models, reads the exact
# MAGIC feature list expected by the boosters, scores 2025-W27..W52, and writes
# MAGIC fresh predictions to Delta.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install lightgbm==4.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import os
import sys

sys.path.append("./")

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from pyspark.sql.types import LongType, StringType, StructField, StructType

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

runs = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT],
    filter_string="tags.mlflow.runName = 'train_pipeline'",
    order_by=["start_time DESC"],
    max_results=1,
)
if len(runs) == 0:
    raise RuntimeError("No train_pipeline run found; run 03_train_model.py first.")

best_threshold = float(runs.iloc[0]["params.best_zero_threshold"])
print(f"Zero threshold: {best_threshold}")

# COMMAND ----------

client = mlflow.tracking.MlflowClient()


def latest_version(name: str) -> str:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No version found for model {name}")
    return str(max(int(v.version) for v in versions))


def best_iteration(model):
    iteration = getattr(model, "best_iteration", None)
    if iteration is None or iteration <= 0:
        return None
    return iteration


v_zero = latest_version(MLFLOW_MODEL_NAME_ZERO)
v_qty = latest_version(MLFLOW_MODEL_NAME_QTY)

model_zero = mlflow.lightgbm.load_model(f"models:/{MLFLOW_MODEL_NAME_ZERO}/{v_zero}")
model_qty = mlflow.lightgbm.load_model(f"models:/{MLFLOW_MODEL_NAME_QTY}/{v_qty}")

model_features = list(model_zero.feature_name())
qty_features = list(model_qty.feature_name())
if model_features != qty_features:
    raise RuntimeError("Zero classifier and quantity regressor do not use the same feature list.")

selected_cat_features = [c for c in FEATURES_CATEGORICAL if c in model_features]
print(f"Loaded zero_classifier v{v_zero}, qty_regressor v{v_qty}")
print(f"Model expects {len(model_features)} features ({len(selected_cat_features)} categorical).")

# COMMAND ----------

test_sdf = spark.table(TBL_GOLD_TEST)
metadata_cols = ["semaine", "code_agence", "code_article", "week_id", "is_dead_pair"]
fallback_cols = [
    "pair_mean_lag26", "pair_median_lag26", "pair_mean",
    "band_mean_26_52", "band_std_26_52", "band_min_26_52",
    "band_max_26_52", "band_zero_rate_26_52",
    "band_active_count_26_52", "band_nonzero_mean_26_52",
    "band_cv_26_52",
]
extra_cols = [c for c in fallback_cols if c in test_sdf.columns and c not in model_features]
needed_cols = metadata_cols + model_features + extra_cols
missing = [c for c in needed_cols if c not in test_sdf.columns]
if missing:
    raise RuntimeError(f"gold_test_features is missing columns required for inference: {missing[:20]}")

test_df = test_sdf.select(*needed_cols).toPandas()
print(f"Test rows: {len(test_df):,}")

# COMMAND ----------

def fill_from_candidates(df: pd.DataFrame, col: str, candidates):
    if col not in df.columns or not df[col].isna().any():
        return
    for candidate in candidates:
        if candidate in df.columns:
            df[col] = df[col].fillna(df[candidate])
            if not df[col].isna().any():
                break


# Defensive fallbacks for old selected models that may still contain short-lag
# or rolling columns. The new selector should mostly avoid them.
for c in model_features:
    if c.startswith("lag_"):
        lag_n = int(c.split("_")[1])
        if lag_n < 26:
            fill_from_candidates(test_df, c, ["pair_mean_lag26", "pair_mean", "band_mean_26_52"])
    elif c.startswith("roll_mean_"):
        fill_from_candidates(test_df, c, ["band_mean_26_52", "pair_mean_lag26", "pair_mean"])
    elif c.startswith("roll_median_"):
        fill_from_candidates(test_df, c, ["pair_median_lag26", "pair_mean_lag26", "pair_mean"])
    elif c.startswith("roll_std_"):
        fill_from_candidates(test_df, c, ["band_std_26_52"])
    elif c.startswith("roll_min_"):
        fill_from_candidates(test_df, c, ["band_min_26_52"])
    elif c.startswith("roll_max_"):
        fill_from_candidates(test_df, c, ["band_max_26_52"])
    elif c.startswith("roll_sum_"):
        fill_from_candidates(test_df, c, ["band_mean_26_52", "pair_mean_lag26", "pair_mean"])
    elif c.startswith("roll_zero_rate_"):
        fill_from_candidates(test_df, c, ["band_zero_rate_26_52"])
    elif c.startswith("roll_active_count_"):
        fill_from_candidates(test_df, c, ["band_active_count_26_52"])
    elif c.startswith("roll_nonzero_mean_"):
        fill_from_candidates(test_df, c, ["band_nonzero_mean_26_52", "pair_mean_lag26", "pair_mean"])
    elif c.startswith("roll_cv_"):
        fill_from_candidates(test_df, c, ["band_cv_26_52"])

X_test = test_df.loc[:, model_features].copy()
for c in model_features:
    if c in selected_cat_features:
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(-1).astype(np.int32)
    else:
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").astype(np.float32)

# COMMAND ----------

p_zero = model_zero.predict(X_test, num_iteration=best_iteration(model_zero))
qty_raw = model_qty.predict(X_test, num_iteration=best_iteration(model_qty))
qty = np.clip(qty_raw, 0.0, None)

final = np.where(p_zero > best_threshold, 0.0, qty)
final = np.where(test_df["is_dead_pair"].fillna(0).astype(int).to_numpy() == 1, 0.0, final)
final_int = np.clip(np.round(final), 0, None).astype(np.int64)

print(f"Predictions > 0: {(final_int > 0).sum():,} / {len(final_int):,}")
print(f"Mean predicted : {final_int.mean():.3f}")

# COMMAND ----------

out_df = test_df[["semaine", "code_agence", "code_article"]].copy()
out_df["code_agence"] = pd.to_numeric(out_df["code_agence"], errors="coerce").astype(np.int64)
out_df["code_article"] = pd.to_numeric(out_df["code_article"], errors="coerce").astype(np.int64)
out_df["quantite"] = final_int

schema = StructType([
    StructField("semaine", StringType()),
    StructField("code_agence", LongType()),
    StructField("code_article", LongType()),
    StructField("quantite", LongType()),
])
out_sdf = spark.createDataFrame(out_df, schema=schema)

(
    out_sdf.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TBL_PREDICTIONS_FINAL)
)
print(f"Wrote final predictions to {TBL_PREDICTIONS_FINAL}")

try:
    (
        out_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TABLE_PREDICTIONS)
    )
    print(f"Wrote submission table to {TABLE_PREDICTIONS}")
except Exception as exc:
    print(f"Could not write submission table {TABLE_PREDICTIONS}: {exc}")
    print(f"Fallback table is available at {TBL_PREDICTIONS_FINAL}")

# COMMAND ----------

with mlflow.start_run(run_name="inference"):
    mlflow.log_param("zero_threshold", best_threshold)
    mlflow.log_param("zero_clf_version", v_zero)
    mlflow.log_param("qty_reg_version", v_qty)
    mlflow.log_param("n_model_features", len(model_features))
    mlflow.log_metric("n_test_rows", len(out_df))
    mlflow.log_metric("n_positive_preds", int((final_int > 0).sum()))
    mlflow.log_metric("mean_pred", float(final_int.mean()))

# COMMAND ----------

display(out_sdf)
