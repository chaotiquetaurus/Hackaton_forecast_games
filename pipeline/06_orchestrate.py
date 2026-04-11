# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Orchestrate (ML side)
# MAGIC
# MAGIC This notebook chains the **ML** notebooks only: training, evaluation,
# MAGIC and inference. The data side (cleaning + feature engineering) is now
# MAGIC handled by the **Lakeflow (DLT) pipeline** defined in
# MAGIC `pipeline_lakeflow.py`.
# MAGIC
# MAGIC **You must run the Lakeflow pipeline first** — either from the
# MAGIC Databricks Pipelines UI, or as the first task of a Job (see
# MAGIC `workflow.json`). This notebook assumes the gold tables are already
# MAGIC populated:
# MAGIC
# MAGIC - `workspace.default.gold_train`
# MAGIC - `workspace.default.gold_validation`
# MAGIC - `workspace.default.gold_internal_test`
# MAGIC - `workspace.default.gold_test_features` (final-inference horizon)
# MAGIC - `workspace.default.silver_ventes` (used by 04 for baselines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick sanity check: the gold tables must exist and be non-empty

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

for tbl in [
    TBL_SILVER_VENTES,
    TBL_GOLD_TRAIN,
    TBL_GOLD_VAL,
    TBL_GOLD_INTERNAL_TEST,
    TBL_GOLD_FINAL_INFERENCE,
]:
    n = spark.table(tbl).count()
    assert n > 0, f"Lakeflow table {tbl} is empty — run pipeline_lakeflow.py first."
    print(f"{tbl:<50} {n:>12,} rows")

# COMMAND ----------

# MAGIC %run ./03_train_model

# COMMAND ----------

# MAGIC %run ./04_evaluate

# COMMAND ----------

# MAGIC %run ./05_inference

# COMMAND ----------

print("=" * 60)
print("ML pipeline finished. See MLflow experiment:")
print(f"  {MLFLOW_EXPERIMENT}")
print("=" * 60)
