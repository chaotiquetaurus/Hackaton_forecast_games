# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Evaluate
# MAGIC
# MAGIC Reads the validation predictions persisted by 03_train_model.py and
# MAGIC reports:
# MAGIC - **Global WAPE** on validation 2024-W27..W52 and internal test 2025-W01..W26
# MAGIC - WAPE broken down by agency, top-20 articles, week, and specialty
# MAGIC - Comparison against three baselines (lag-52, global mean, blend)
# MAGIC - Zero-classifier confusion matrix
# MAGIC
# MAGIC Everything lands in MLflow as metrics + artifacts, and a concise
# MAGIC summary table is printed at the end of the notebook.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import sys
sys.path.append("./")

import numpy as np
import pandas as pd
import mlflow
from pyspark.sql import functions as F

from src.utils import wape_numpy, wape_spark

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load validation predictions and join reference data

# COMMAND ----------

val_sdf = spark.table(TBL_VAL_PREDICTIONS)
internal_test_sdf = spark.table(TBL_INTERNAL_TEST_PREDICTIONS)
articles = spark.table(TBL_ARTICLES).select("code_agence", "code_article",
    *[c for c in ["specialite"] if c in spark.table(TBL_ARTICLES).columns]
)

val_enriched = val_sdf.join(articles, ["code_agence", "code_article"], "left")
internal_test_enriched = internal_test_sdf.join(articles, ["code_agence", "code_article"], "left")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Global WAPE and baselines
# MAGIC
# MAGIC We re-compute the same three baselines on the validation period so
# MAGIC that the report shows the model's lift above the simplest possible
# MAGIC predictors:
# MAGIC   - `naive_lag52`: copy the value from 52 weeks earlier.
# MAGIC   - `global_mean`: predict the pair's training-period mean.
# MAGIC   - `blend_seasonal`: 0.6 * lag_52 + 0.4 * pair_mean (the EDA report
# MAGIC     cites 1.259 for this family of blends).

# COMMAND ----------

# Baselines are computed from the silver cleaned ventes produced by DLT,
# not from the gold train split, because gold_train drops the raw columns
# we need (annee, num_sem) in some older pipeline runs. silver_ventes is
# the exact cleaned train table.
cleaned = spark.table(TBL_SILVER_VENTES)
train_only = cleaned.filter(F.col("week_id") < VAL_START_WEEK_ID)

pair_mean_df = (
    train_only
    .groupBy("code_agence", "code_article")
    .agg(F.avg("quantite").alias("pair_mean_train"))
)

# lag_52: for a row (annee=2025, num_sem=S), take the row (annee=2024, num_sem=S).
val_cols = val_sdf.select("semaine", "code_agence", "code_article", "quantite")

lag52_src = (
    train_only
    .withColumn("semaine_plus1y",
        F.concat((F.col("annee") + 1).cast("string"), F.lit("-"),
                 F.lpad(F.col("num_sem").cast("string"), 2, "0")))
    .select(
        F.col("semaine_plus1y").alias("semaine"),
        "code_agence",
        "code_article",
        F.col("quantite").alias("lag52_pred"),
    )
)

val_base = (
    val_cols
    .join(lag52_src, ["semaine", "code_agence", "code_article"], "left")
    .join(pair_mean_df, ["code_agence", "code_article"], "left")
    .withColumn("lag52_pred", F.coalesce(F.col("lag52_pred"), F.lit(0.0)))
    .withColumn("pair_mean_train", F.coalesce(F.col("pair_mean_train"), F.lit(0.0)))
    .withColumn("blend_pred", F.col("lag52_pred") * F.lit(0.6) + F.col("pair_mean_train") * F.lit(0.4))
)

baselines = {
    "naive_lag52": wape_spark(val_base, "quantite", "lag52_pred"),
    "global_mean": wape_spark(val_base, "quantite", "pair_mean_train"),
    "blend_seasonal": wape_spark(val_base, "quantite", "blend_pred"),
}

# Our model's WAPE on the same set, plus the internal test held out from tuning.
model_wape = wape_spark(val_sdf, "quantite", "prediction")
internal_test_wape = wape_spark(internal_test_sdf, "quantite", "prediction")

print("=== WAPE summary ===")
summary_rows = [{"split": "validation", "model": k, "wape": v} for k, v in baselines.items()]
summary_rows.append({"split": "validation", "model": "two_stage_lgbm", "wape": model_wape})
summary_rows.append({"split": "internal_test", "model": "two_stage_lgbm", "wape": internal_test_wape})
summary_df = pd.DataFrame(summary_rows).sort_values(["split", "wape"])
print(summary_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. WAPE broken down by slice

# COMMAND ----------

val_pd = val_enriched.toPandas()
internal_test_pd = internal_test_enriched.toPandas()

def wape_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    grp = (
        df.groupby(key)
        .apply(lambda g: pd.Series({
            "wape": wape_numpy(g["quantite"].values, g["prediction"].values),
            "n": len(g),
            "sum_actual": float(g["quantite"].sum()),
        }))
        .reset_index()
    )
    return grp.sort_values("wape", ascending=False)

wape_by_agency = wape_by(val_pd, "code_agence")
wape_by_week = wape_by(val_pd, "semaine")
test_wape_by_week = wape_by(internal_test_pd, "semaine")

top_articles = (
    val_pd.groupby("code_article")["quantite"].sum().nlargest(20).index
)
wape_top_articles = wape_by(val_pd[val_pd["code_article"].isin(top_articles)], "code_article")

if "specialite" in val_pd.columns:
    wape_by_specialty = wape_by(val_pd, "specialite")
else:
    wape_by_specialty = pd.DataFrame(columns=["specialite", "wape", "n", "sum_actual"])

# Zero vs non-zero slice.
wape_zero = wape_numpy(
    val_pd.loc[val_pd["quantite"] == 0, "quantite"].values,
    val_pd.loc[val_pd["quantite"] == 0, "prediction"].values,
)
wape_nonzero = wape_numpy(
    val_pd.loc[val_pd["quantite"] > 0, "quantite"].values,
    val_pd.loc[val_pd["quantite"] > 0, "prediction"].values,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Zero classifier confusion matrix (at the adaptive per-row threshold)

# COMMAND ----------

# 03_train_model writes an effective per-row `threshold` column alongside
# `p_zero`. We just read it back — no need to query MLflow for params.
if "threshold" not in val_pd.columns:
    raise RuntimeError(
        "val_predictions table is missing the `threshold` column. "
        "Re-run 03_train_model.py to regenerate it."
    )

y_true_zero = (val_pd["quantite"] == 0).astype(int).values
y_pred_zero = (val_pd["p_zero"].values > val_pd["threshold"].values).astype(int)

threshold = float(val_pd["threshold"].mean())  # mean effective threshold, for the print
print(f"Using per-row adaptive thresholds (mean = {threshold:.3f})")

tp = int(((y_pred_zero == 1) & (y_true_zero == 1)).sum())
tn = int(((y_pred_zero == 0) & (y_true_zero == 0)).sum())
fp = int(((y_pred_zero == 1) & (y_true_zero == 0)).sum())
fn = int(((y_pred_zero == 0) & (y_true_zero == 1)).sum())

precision_zero = tp / (tp + fp + 1e-9)
recall_zero = tp / (tp + fn + 1e-9)

print(f"Zero clf @ thr={threshold}: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"   precision={precision_zero:.3f}  recall={recall_zero:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log everything to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="evaluation"):
    mlflow.log_metric("val_wape_model", model_wape)
    mlflow.log_metric("internal_test_wape_model", internal_test_wape)
    for name, v in baselines.items():
        mlflow.log_metric(f"val_wape_{name}", v)
    mlflow.log_metric("val_wape_zero_slice", wape_zero)
    mlflow.log_metric("val_wape_nonzero_slice", wape_nonzero)
    mlflow.log_metrics({
        "zero_precision": precision_zero,
        "zero_recall": recall_zero,
        "zero_tp": tp,
        "zero_tn": tn,
        "zero_fp": fp,
        "zero_fn": fn,
    })

    summary_df.to_csv("/tmp/wape_summary.csv", index=False)
    wape_by_agency.to_csv("/tmp/wape_by_agency.csv", index=False)
    wape_by_week.to_csv("/tmp/wape_by_week.csv", index=False)
    test_wape_by_week.to_csv("/tmp/internal_test_wape_by_week.csv", index=False)
    wape_top_articles.to_csv("/tmp/wape_top_articles.csv", index=False)
    wape_by_specialty.to_csv("/tmp/wape_by_specialty.csv", index=False)
    for f in [
        "/tmp/wape_summary.csv",
        "/tmp/wape_by_agency.csv",
        "/tmp/wape_by_week.csv",
        "/tmp/internal_test_wape_by_week.csv",
        "/tmp/wape_top_articles.csv",
        "/tmp/wape_by_specialty.csv",
    ]:
        mlflow.log_artifact(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Final printed summary

# COMMAND ----------

print("=" * 60)
print("             SGDB 2026 — VALIDATION REPORT")
print("=" * 60)
print(summary_df.to_string(index=False))
print()
print(f"Internal test WAPE : {internal_test_wape:.4f}")
print()
print(f"Zero-slice WAPE     : {wape_zero:.4f}")
print(f"Non-zero slice WAPE : {wape_nonzero:.4f}")
print()
print("--- WAPE per agency (worst 5) ---")
print(wape_by_agency.head(5).to_string(index=False))
print()
print("--- WAPE per week ---")
print(wape_by_week.to_string(index=False))
print()
print("--- Internal test WAPE per week ---")
print(test_wape_by_week.to_string(index=False))
print()
print(f"Zero-clf precision / recall @ adaptive thr (mean={threshold:.3f}): "
      f"{precision_zero:.3f} / {recall_zero:.3f}")
