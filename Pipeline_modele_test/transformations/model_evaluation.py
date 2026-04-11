from pyspark import pipelines as dp
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Evaluation: WAPE on a time-based train/validation split
# WAPE = sum(|actual - predicted|) / sum(|actual|)
# ---------------------------------------------------------------------------

@dp.materialized_view(comment="WAPE score of the dummy model (train < 2025, val >= 2025)")
def wape_score():
    df = spark.read.table("features_train")

    # Split: use pre-2025 to compute averages, evaluate on 2025+
    train_data = df.filter(F.col("annee") < 2025)
    val_data = df.filter(F.col("annee") >= 2025)

    # Same dummy model logic: average quantity per (agency, article)
    avg_per_pair = (
        train_data.groupBy("code_agence", "code_article")
        .agg(F.avg("quantite").alias("prediction"))
    )

    val_with_preds = (
        val_data
        .join(avg_per_pair, ["code_agence", "code_article"], "left")
        .withColumn("prediction", F.coalesce(F.col("prediction"), F.lit(0.0)))
    )

    # WAPE = sum(|actual - predicted|) / sum(|actual|)
    return val_with_preds.agg(
        (
            F.sum(F.abs(F.col("quantite") - F.col("prediction")))
            / F.sum(F.abs(F.col("quantite")))
        ).alias("wape"),
        F.count("*").alias("n_validation_samples"),
    )
