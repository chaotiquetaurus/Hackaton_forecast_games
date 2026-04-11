from pyspark import pipelines as dp
from config import (
    TBL_TRAIN, TBL_TEST, TBL_AGENCE, TBL_ARTICLES, TBL_FACTURATION,
)


@dp.materialized_view(
    name="bronze_ventes",
    comment="Raw sales history (histo_ventes_train) — exact passthrough.",
)
def bronze_ventes():
    return spark.read.table(TBL_TRAIN)


@dp.materialized_view(
    name="bronze_ventes_test",
    comment="Raw test rows to predict (histo_ventes_test) — exact passthrough.",
)
def bronze_ventes_test():
    return spark.read.table(TBL_TEST)


@dp.materialized_view(
    name="bronze_agences",
    comment="Agency reference data (donnees_agence).",
)
def bronze_agences():
    return spark.read.table(TBL_AGENCE)


@dp.materialized_view(
    name="bronze_articles",
    comment="Article reference data (donnees_articles) — keyed by (agency, article).",
)
def bronze_articles():
    return spark.read.table(TBL_ARTICLES)


@dp.materialized_view(
    name="bronze_facturation",
    comment="Monthly billing aggregates (donnees_facturation).",
)
def bronze_facturation():
    return spark.read.table(TBL_FACTURATION)
