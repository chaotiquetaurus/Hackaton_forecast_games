# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Feature Engineering + LightGBM — Hackathon SGDB 2026
# MAGIC Pipeline complet : features → train → validate → predict → submit

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

!pip install lightgbm

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings("ignore")

TEAM_NAME = "mon_equipe"  # <-- CHANGE ICI

# COMMAND ----------

# Chargement
train_raw = spark.table("workspace.default.histo_ventes_train")
test_raw  = spark.table("workspace.default.histo_ventes_test")
agences   = spark.table("workspace.default.donnees_agence")
articles  = spark.table("workspace.default.donnees_articles")
factu     = spark.table("workspace.default.donnees_facturation")

print("✅ Tables chargées")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Préparation de base

# COMMAND ----------

# Extraire année et numéro de semaine
def add_time_cols(df):
    return (
        df
        .withColumn("annee", F.split("semaine", "-")[0].cast("int"))
        .withColumn("num_semaine", F.split("semaine", "-")[1].cast("int"))
        # Créer un index semaine continu pour le tri et les lags
        .withColumn("sem_idx", F.split("semaine", "-")[0].cast("int") * 52 + F.split("semaine", "-")[1].cast("int"))
    )

train_sp = add_time_cols(train_raw)
test_sp  = add_time_cols(test_raw.withColumn("quantite", F.lit(None).cast("long")))

# Union pour construire les features sur tout le timeline
full = train_sp.unionByName(test_sp)

print(f"Train: {train_sp.count():,}  |  Test: {test_sp.count():,}  |  Full: {full.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering (PySpark)

# COMMAND ----------

# ---------------------------------------------------------------
# 2a. LAGS ET ROLLING (sur la paire agence × article)
# ---------------------------------------------------------------
w = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx")

# Lags simples — SHIFT(1) implicite car on prend des lags >= 1
for lag_n in [1, 2, 4, 8, 12, 26, 52]:
    full = full.withColumn(f"lag_{lag_n}", F.lag("quantite", lag_n).over(w))

# Rolling moyennes — calculées sur le lag_1 pour éviter le leakage
# On fait rolling sur les valeurs déjà shiftées
for win_size in [4, 12, 26, 52]:
    w_roll = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(-win_size, -1)
    full = full.withColumn(f"roll_mean_{win_size}", F.avg("quantite").over(w_roll))
    full = full.withColumn(f"roll_std_{win_size}", F.stddev("quantite").over(w_roll))
    full = full.withColumn(f"roll_max_{win_size}", F.max("quantite").over(w_roll))
    full = full.withColumn(f"roll_min_{win_size}", F.min("quantite").over(w_roll))

# Rolling taux de zéros
for win_size in [4, 12, 52]:
    w_roll = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(-win_size, -1)
    full = full.withColumn(
        f"zero_rate_{win_size}",
        F.avg(F.when(F.col("quantite") == 0, 1.0).otherwise(0.0)).over(w_roll)
    )

print("✅ Lags + Rolling OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2b. TENDANCES
# ---------------------------------------------------------------
full = full.withColumn(
    "trend_4_12",
    F.when(F.col("roll_mean_12") > 0, F.col("roll_mean_4") / F.col("roll_mean_12")).otherwise(None)
)
full = full.withColumn(
    "trend_4_52",
    F.when(F.col("roll_mean_52") > 0, F.col("roll_mean_4") / F.col("roll_mean_52")).otherwise(None)
)
full = full.withColumn(
    "yoy_ratio",
    F.when(F.col("lag_52") > 0, F.col("lag_1") / F.col("lag_52")).otherwise(None)
)

print("✅ Tendances OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2c. SAISONNALITE
# ---------------------------------------------------------------
full = full.withColumn("sem_sin", F.sin(2 * np.pi * F.col("num_semaine") / 52))
full = full.withColumn("sem_cos", F.cos(2 * np.pi * F.col("num_semaine") / 52))

print("✅ Saisonnalité OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2d. STATS PAIRE (calculées UNIQUEMENT sur données < semaine courante)
# ---------------------------------------------------------------
w_expanding = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)

full = full.withColumn("pair_expanding_mean", F.avg("quantite").over(w_expanding))
full = full.withColumn("pair_expanding_std", F.stddev("quantite").over(w_expanding))
full = full.withColumn("pair_expanding_max", F.max("quantite").over(w_expanding))
full = full.withColumn("pair_expanding_count", F.count("quantite").over(w_expanding))
full = full.withColumn(
    "pair_expanding_zero_rate",
    F.avg(F.when(F.col("quantite") == 0, 1.0).otherwise(0.0)).over(w_expanding)
)

print("✅ Stats paire expanding OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2e. STATS AGENCE expanding (safe, pas de leakage)
# ---------------------------------------------------------------
w_ag = Window.partitionBy("code_agence").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("agence_expanding_mean", F.avg("quantite").over(w_ag))

# STATS ARTICLE expanding
w_art = Window.partitionBy("code_article").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("article_expanding_mean", F.avg("quantite").over(w_art))

print("✅ Stats agence/article expanding OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2f. ENRICHISSEMENT AGENCES
# ---------------------------------------------------------------
agence_features = agences.select(
    "code_agence",
    F.col("region").alias("ag_region"),
    F.col("secteur").alias("ag_secteur"),
    F.col("departement").alias("ag_departement"),
    F.col("latitude").alias("ag_latitude"),
    F.col("longitude").alias("ag_longitude"),
)

full = full.join(agence_features, "code_agence", "left")

print("✅ Enrichissement agences OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2g. ENRICHISSEMENT ARTICLES (attention : indexé par agence × article)
# ---------------------------------------------------------------
article_features = articles.select(
    "code_agence",
    "code_article",
    F.col("sous_famille").alias("art_sous_famille"),
    F.col("famille").alias("art_famille"),
    F.col("marque").alias("art_marque"),
    F.col("specialite").alias("art_specialite"),
    F.col("unite_vente").alias("art_unite_vente"),
    F.col("article_mdd").alias("art_mdd"),
    F.col("Gamme").alias("art_gamme"),
)

full = full.join(article_features, ["code_agence", "code_article"], "left")

print("✅ Enrichissement articles OK")

# COMMAND ----------

# ---------------------------------------------------------------
# 2h. ENRICHISSEMENT FACTURATION (mensuel → à agréger par paire)
# ---------------------------------------------------------------
# Calculer des stats globales par paire depuis la facturation
factu_agg = (
    factu
    .groupBy("code_agence", "code_article")
    .agg(
        F.round(F.avg("sum_quantite"), 2).alias("fac_avg_qty"),
        F.round(F.avg("sum_montant"), 2).alias("fac_avg_montant"),
        F.round(F.avg("nb_achats"), 2).alias("fac_avg_nb_achats"),
        F.round(F.sum("nb_achats_par_professionnels") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_pro"),
        F.round(F.sum("nb_ventes_magasins") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_magasin"),
        F.round(F.avg("nb_chantiers"), 2).alias("fac_avg_chantiers"),
        # Prix unitaire moyen
        F.round(F.sum("sum_montant") / (F.sum("sum_quantite") + 0.001), 2).alias("fac_prix_unit"),
    )
)

full = full.join(factu_agg, ["code_agence", "code_article"], "left")

print("✅ Enrichissement facturation OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Conversion Pandas + Encoding

# COMMAND ----------

# Séparer train et test
df = full.toPandas()
print(f"Full shape: {df.shape}")

# Colonnes catégorielles à encoder
cat_cols = ["ag_region", "ag_secteur", "art_famille", "art_sous_famille",
            "art_marque", "art_specialite", "art_unite_vente", "art_gamme"]

# Label encoding
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category").cat.codes

# Séparer train / test
mask_test = df["semaine"].str.startswith("2025-") & (df["num_semaine"] >= 27)
mask_val  = df["semaine"].str.startswith("2025-") & (df["num_semaine"] >= 1) & (df["num_semaine"] <= 26)
mask_train = ~mask_test & ~mask_val

df_train = df[mask_train].copy()
df_val   = df[mask_val].copy()
df_test  = df[mask_test].copy()

print(f"Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  Test: {len(df_test):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Définir les features

# COMMAND ----------

# Features à exclure
exclude = ["semaine", "code_agence", "code_article", "quantite", "annee", "sem_idx"]

feature_cols = [c for c in df.columns if c not in exclude]
print(f"Nb features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Entraînement LightGBM

# COMMAND ----------

# ---------------------------------------------------------------
# 5a. WAPE metric
# ---------------------------------------------------------------
def wape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return np.sum(np.abs(y_pred - y_true)) / denom

# ---------------------------------------------------------------
# 5b. Train sur train, valider sur val
# ---------------------------------------------------------------
X_train = df_train[feature_cols].values
y_train = df_train["quantite"].values
X_val   = df_val[feature_cols].values
y_val   = df_val["quantite"].values

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain, free_raw_data=False)

params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
}

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100),
    ],
)

print(f"\n✅ Best iteration: {model.best_iteration}")

# COMMAND ----------

# ---------------------------------------------------------------
# 5c. Évaluation WAPE sur validation
# ---------------------------------------------------------------
y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_val_pred = np.clip(y_val_pred, 0, None)  # pas de négatifs
y_val_pred_rounded = np.round(y_val_pred).astype(int)

val_wape = wape(y_val, y_val_pred_rounded)
print(f"\n{'='*50}")
print(f"  WAPE validation (S01-S26 2025) : {val_wape:.4f}")
print(f"{'='*50}")

# Comparaison baseline
print(f"  Baseline saisonnière  : ~1.387")
print(f"  Baseline blend        : ~1.259")
print(f"  Notre modèle          : {val_wape:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Importance

# COMMAND ----------

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)

print("🏆 Top 30 features par importance (gain) :")
print(importance.head(30).to_string(index=False))

# Features inutiles (importance = 0)
zero_imp = importance[importance["importance"] == 0]
if len(zero_imp) > 0:
    print(f"\n⚠️  {len(zero_imp)} features avec importance 0 :")
    print(zero_imp["feature"].tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Re-train sur TOUT (train + val) et prédire le test

# COMMAND ----------

# Combiner train + val pour le modèle final
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

dfull = lgb.Dataset(X_full, label=y_full, feature_name=feature_cols, free_raw_data=False)

model_final = lgb.train(
    params,
    dfull,
    num_boost_round=model.best_iteration,  # Utiliser le best iteration du modèle validé
)

print("✅ Modèle final entraîné sur train+val")

# COMMAND ----------

# Prédiction test
X_test = df_test[feature_cols].values
y_test_pred = model_final.predict(X_test)
y_test_pred = np.clip(y_test_pred, 0, None)
y_test_pred = np.round(y_test_pred).astype(int)

df_test["quantite_pred"] = y_test_pred

print(f"Prédictions test : {len(df_test):,} lignes")
print(f"Min pred: {y_test_pred.min()}  |  Max pred: {y_test_pred.max()}")
print(f"Mean pred: {y_test_pred.mean():.2f}  |  % zéros: {(y_test_pred == 0).mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sauvegarder les prédictions

# COMMAND ----------

# Format attendu : semaine, code_agence, code_article, quantite
submission = df_test[["semaine", "code_agence", "code_article"]].copy()
submission["quantite"] = y_test_pred

# Vérifications
n_expected = 272344
n_actual = len(submission)
print(f"Lignes attendues : {n_expected:,}")
print(f"Lignes produites : {n_actual:,}")
assert n_actual == n_expected, f"⚠️ ERREUR : {n_actual} != {n_expected} lignes !"

# Convertir en Spark et sauvegarder
submission_sp = spark.createDataFrame(submission)
submission_sp = (
    submission_sp
    .withColumn("code_agence", F.col("code_agence").cast("long"))
    .withColumn("code_article", F.col("code_article").cast("long"))
    .withColumn("quantite", F.col("quantite").cast("long"))
)

table_name = f"workspace.default.predictions_equipe_{TEAM_NAME}"
submission_sp.write.mode("overwrite").saveAsTable(table_name)

print(f"\n✅ Prédictions sauvegardées dans : {table_name}")
print(f"   → Allez sur l'app pour calculer votre WAPE officiel !")

# COMMAND ----------


