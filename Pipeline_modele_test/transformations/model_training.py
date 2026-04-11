from pyspark import pipelines as dp
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Dummy model: predict the historical mean quantity per (agency, article)
# ---------------------------------------------------------------------------

@dp.materialized_view(comment="Dummy predictions: historical average quantity per agency-article pair")
def predictions():
    train = spark.read.table("features_train")
    test = spark.read.table("features_test")

    avg_per_pair = (
        train.groupBy("code_agence", "code_article")
        .agg(F.avg("quantite").alias("predicted_quantite"))
    )

    return (
        test
        .join(avg_per_pair, ["code_agence", "code_article"], "left")
        .withColumn("predicted_quantite",
                    F.round(F.coalesce(F.col("predicted_quantite"), F.lit(0.0)), 2))
        .select("semaine", "code_agence", "code_article", "predicted_quantite")
    )
