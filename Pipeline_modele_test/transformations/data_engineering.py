from pyspark import pipelines as dp
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Data engineering: read and prepare the raw sales data
# ---------------------------------------------------------------------------

@dp.materialized_view(comment="Training sales data with parsed time features")
def features_train():
    return (
        spark.read.table("workspace.default.histo_ventes_train")
        .withColumn("annee", F.split(F.col("semaine"), "-").getItem(0).cast("int"))
        .withColumn("num_semaine", F.split(F.col("semaine"), "-").getItem(1).cast("int"))
    )


@dp.materialized_view(comment="Test sales data with parsed time features")
def features_test():
    return (
        spark.read.table("workspace.default.histo_ventes_test")
        .withColumn("annee", F.split(F.col("semaine"), "-").getItem(0).cast("int"))
        .withColumn("num_semaine", F.split(F.col("semaine"), "-").getItem(1).cast("int"))
    )
