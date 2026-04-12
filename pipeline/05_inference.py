# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Inference
# MAGIC
# MAGIC Applies the two-stage model (pulled from the MLflow Model Registry)
# MAGIC to the test period (2025-W27 .. 2025-W52) and writes the final
# MAGIC predictions to Delta + a CSV for the hackathon submission.
# MAGIC
# MAGIC **Inputs:**
# MAGIC - `workspace.default.feature_table` (already contains test rows with
# MAGIC   features but `quantite = null`)
# MAGIC - The latest versions of both registered models.
# MAGIC - The best zero threshold, pulled from the latest `train_pipeline`
# MAGIC   MLflow run.
# MAGIC
# MAGIC **Outputs:**
# MAGIC - `workspace.default.predictions_final` — Delta table with columns
# MAGIC   (semaine, code_agence, code_article, quantite).
# MAGIC - The hackathon submission table `predictions_equipe_<team>`
# MAGIC   (overwritten with the same contents).
# MAGIC - A CSV file in `/dbfs/FileStore/sgdb2026_submission.csv`.
# MAGIC
# MAGIC **Lag handling at inference:** lags 1..25 are not available when
# MAGIC predicting W27..W52 because they would require target values from
# MAGIC inside the prediction window. We impute them with the pair's
# MAGIC expanding mean (`pair_mean`, itself safe), which is the same fallback
# MAGIC LightGBM would pick with missing-value splits anyway — being explicit
# MAGIC about it makes the behaviour reproducible.

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
import mlflow
import mlflow.lightgbm
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

from src.utils import prepare_features

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the tuned threshold from the latest training run

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

# MAGIC %md
# MAGIC ## 2. Load both models from the registry

# COMMAND ----------

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
# MAGIC ## 3. Pull the feature rows for the test period
# MAGIC
# MAGIC The Lakeflow pipeline already materialises `gold_test_features` with
# MAGIC exactly the test period (2025-W27..W52) and the same features as
# MAGIC `gold_train` / `gold_validation`. We just project and ship to pandas.

# COMMAND ----------

# Exclude derived features computed by prepare_features() in pandas.
_DERIVED_FEATURES = {"detrended_lag52", "demand_profile"}
feature_cols = (
    ["semaine", "code_agence", "code_article", "week_id", "is_dead_pair"]
    + [f for f in FEATURES if f not in _DERIVED_FEATURES]
)

test_features_sdf = spark.table(TBL_GOLD_TEST).select(*feature_cols)

test_df = test_features_sdf.toPandas()
print(f"Test rows: {len(test_df):,}   (expected 272 344)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature preparation (clipping + derived features)
# MAGIC
# MAGIC All features are now inference-safe (lags >= 26, band_* windows, _lag26
# MAGIC pair stats), so the old short-lag imputation is no longer needed.
# MAGIC `prepare_features` clips extreme ratio/CV values and adds the two
# MAGIC derived features (`detrended_lag52`, `demand_profile`) — same function
# MAGIC used at training time for consistency.

# COMMAND ----------

test_df = prepare_features(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Score both stages and combine

# COMMAND ----------

X_test = test_df[FEATURES].copy()
for c in FEATURES_CATEGORICAL:
    if c in X_test.columns:
        X_test[c] = X_test[c].astype("category")

p_zero = model_zero.predict(X_test, num_iteration=model_zero.best_iteration)
qty_raw = model_qty.predict(X_test, num_iteration=model_qty.best_iteration)
qty = np.clip(qty_raw, 0.0, None)

final = np.where(p_zero > best_threshold, 0.0, qty)

# Dead pairs are forced to zero regardless — no reason to believe they will
# suddenly come back to life in W27..W52.
final = np.where(test_df["is_dead_pair"].values == 1, 0.0, final)

# Round to non-negative integers — the submission format expects LongType.
final_int = np.clip(np.round(final), 0, None).astype(np.int64)

print(f"Predictions > 0: {(final_int > 0).sum():,}  /  total: {len(final_int):,}")
print(f"Mean predicted : {final_int.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### eNZO Test

# COMMAND ----------


NOM_EQUIPE = "telecacaton"   # ← remplacez par le nom de votre équipe

# Ne touchez pas au reste
TABLE_PREDICTIONS = f"workspace.default.predictions_equipe_{NOM_EQUIPE}"
print(f"Votre table de prédictions : {TABLE_PREDICTIONS}")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write final predictions

# COMMAND ----------

import base64

out_sdf = spark.table("workspace.default.predictions_final")
print(f"{out_sdf.count():,} rows")

# Write full CSV for download
out_pd = out_sdf.toPandas()
csv_bytes = out_pd.to_csv(index=False).encode("utf-8")
b64 = base64.b64encode(csv_bytes).decode("utf-8")

displayHTML(
    f'<a download="predictions_final.csv" '
    f'href="data:text/csv;base64,{b64}" '
    f'style="font-size:16px">⬇ Download all {len(out_pd):,} rows as CSV</a>'
)

display(out_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. CSV export for manual submission

# COMMAND ----------

print("Predictions already saved to Delta table — CSV export skipped (DBFS not available on serverless).")
print(f"Query your predictions with: SELECT * FROM {TABLE_PREDICTIONS}")

# COMMAND ----------

with mlflow.start_run(run_name="inference"):
    mlflow.log_param("zero_threshold", best_threshold)
    mlflow.log_param("zero_clf_version", v_zero)
    mlflow.log_param("qty_reg_version", v_qty)
    mlflow.log_metric("n_test_rows", len(out_df))
    mlflow.log_metric("n_positive_preds", int((final_int > 0).sum()))
    mlflow.log_metric("mean_pred", float(final_int.mean()))

# COMMAND ----------

# Leaderboard submission skipped — no permissions on predictions_equipe table
print("Download the CSV from the table above instead.")
