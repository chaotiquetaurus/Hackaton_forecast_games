# Databricks notebook source
# MAGIC %md
# MAGIC # 🧪 Feature Factory + Sélection — Hackathon SGDB 2026
# MAGIC
# MAGIC **Stratégie :** Générer ~150+ features → sélection automatique → modèle final lean
# MAGIC
# MAGIC Pipeline :
# MAGIC 1. Feature engineering massif (PySpark)
# MAGIC 2. Conversion Pandas
# MAGIC 3. Feature selection (guide méthodologique)
# MAGIC 4. Training final Tweedie + WAPE

# COMMAND ----------

!pip install lightgbm

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import lightgbm as lgb
import json, os
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

def add_time_cols(df):
    return (
        df
        .withColumn("annee", F.split("semaine", "-")[0].cast("int"))
        .withColumn("num_sem", F.split("semaine", "-")[1].cast("int"))
        .withColumn("sem_idx", F.split("semaine", "-")[0].cast("int") * 100 + F.split("semaine", "-")[1].cast("int"))
    )

train_sp = add_time_cols(train_raw).withColumn("is_test", F.lit(0))
test_sp  = add_time_cols(test_raw.withColumn("quantite", F.lit(None).cast("long"))).withColumn("is_test", F.lit(1))
full = train_sp.unionByName(test_sp)

print(f"✅ Full: {full.count():,} lignes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. FEATURE FACTORY (~150+ features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1a. Lags (14 features)

# COMMAND ----------

w = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx")

# Lags classiques
for lag_n in [1, 2, 3, 4, 6, 8, 10, 13, 17, 20, 26, 39, 44, 52]:
    full = full.withColumn(f"lag_{lag_n}", F.lag("quantite", lag_n).over(w))

print("✅ 14 lags")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1b. Rolling stats sur la paire (40 features)

# COMMAND ----------

for win_size in [4, 8, 13, 26, 52]:
    w_roll = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(-win_size, -1)
    full = full.withColumn(f"roll_mean_{win_size}",   F.avg("quantite").over(w_roll))
    full = full.withColumn(f"roll_std_{win_size}",    F.stddev("quantite").over(w_roll))
    full = full.withColumn(f"roll_max_{win_size}",    F.max("quantite").over(w_roll))
    full = full.withColumn(f"roll_min_{win_size}",    F.min("quantite").over(w_roll))
    full = full.withColumn(f"roll_median_{win_size}", F.percentile_approx("quantite", 0.5).over(w_roll))
    # Coefficient de variation
    full = full.withColumn(
        f"roll_cv_{win_size}",
        F.when(F.col(f"roll_mean_{win_size}") > 0,
               F.col(f"roll_std_{win_size}") / F.col(f"roll_mean_{win_size}")
        ).otherwise(None)
    )
    # Skew proxy : (mean - median) / (std + 1)
    full = full.withColumn(
        f"roll_skew_{win_size}",
        F.when(F.col(f"roll_std_{win_size}") > 0,
               (F.col(f"roll_mean_{win_size}") - F.col(f"roll_median_{win_size}")) / (F.col(f"roll_std_{win_size}") + 1)
        ).otherwise(0)
    )
    # Range
    full = full.withColumn(
        f"roll_range_{win_size}",
        F.col(f"roll_max_{win_size}") - F.col(f"roll_min_{win_size}")
    )

print("✅ 40 rolling stats (8 × 5 fenêtres)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1c. Taux de zéros (6 features)

# COMMAND ----------

for win_size in [4, 8, 13, 26, 39, 52]:
    w_roll = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(-win_size, -1)
    full = full.withColumn(
        f"zero_rate_{win_size}",
        F.avg(F.when(F.col("quantite") == 0, 1.0).otherwise(0.0)).over(w_roll)
    )

print("✅ 6 zero rates")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1d. Diffs et ratios entre lags (10 features)

# COMMAND ----------

# Différences absolues
full = full.withColumn("diff_1_2", F.col("lag_1") - F.col("lag_2"))
full = full.withColumn("diff_1_4", F.col("lag_1") - F.col("lag_4"))
full = full.withColumn("diff_1_52", F.col("lag_1") - F.col("lag_52"))
full = full.withColumn("diff_26_52", F.col("lag_26") - F.col("lag_52"))

# Ratios
full = full.withColumn("ratio_1_4", F.when(F.col("lag_4") > 0, F.col("lag_1") / F.col("lag_4")).otherwise(None))
full = full.withColumn("ratio_1_13", F.when(F.col("lag_13") > 0, F.col("lag_1") / F.col("lag_13")).otherwise(None))
full = full.withColumn("ratio_1_52", F.when(F.col("lag_52") > 0, F.col("lag_1") / F.col("lag_52")).otherwise(None))
full = full.withColumn("ratio_4_13", F.when(F.col("lag_13") > 0, F.col("lag_4") / F.col("lag_13")).otherwise(None))
full = full.withColumn("ratio_4_52", F.when(F.col("lag_52") > 0, F.col("lag_4") / F.col("lag_52")).otherwise(None))
full = full.withColumn("ratio_13_52", F.when(F.col("lag_52") > 0, F.col("lag_13") / F.col("lag_52")).otherwise(None))

print("✅ 10 diffs/ratios lags")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1e. Tendances rolling (6 features)

# COMMAND ----------

full = full.withColumn("trend_4_8",  F.when(F.col("roll_mean_8")  > 0, F.col("roll_mean_4") / F.col("roll_mean_8")).otherwise(None))
full = full.withColumn("trend_4_13", F.when(F.col("roll_mean_13") > 0, F.col("roll_mean_4") / F.col("roll_mean_13")).otherwise(None))
full = full.withColumn("trend_4_26", F.when(F.col("roll_mean_26") > 0, F.col("roll_mean_4") / F.col("roll_mean_26")).otherwise(None))
full = full.withColumn("trend_4_52", F.when(F.col("roll_mean_52") > 0, F.col("roll_mean_4") / F.col("roll_mean_52")).otherwise(None))
full = full.withColumn("trend_8_26", F.when(F.col("roll_mean_26") > 0, F.col("roll_mean_8") / F.col("roll_mean_26")).otherwise(None))
full = full.withColumn("trend_13_52", F.when(F.col("roll_mean_52") > 0, F.col("roll_mean_13") / F.col("roll_mean_52")).otherwise(None))

print("✅ 6 tendances rolling")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1f. Stats expanding paire / agence / article (14 features)

# COMMAND ----------

w_exp = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("pair_exp_mean",      F.avg("quantite").over(w_exp))
full = full.withColumn("pair_exp_median",    F.percentile_approx("quantite", 0.5).over(w_exp))
full = full.withColumn("pair_exp_std",       F.stddev("quantite").over(w_exp))
full = full.withColumn("pair_exp_max",       F.max("quantite").over(w_exp))
full = full.withColumn("pair_exp_count",     F.count("quantite").over(w_exp))
full = full.withColumn("pair_exp_zero_rate", F.avg(F.when(F.col("quantite") == 0, 1.0).otherwise(0.0)).over(w_exp))
# CV expanding
full = full.withColumn("pair_exp_cv",
    F.when(F.col("pair_exp_mean") > 0, F.col("pair_exp_std") / F.col("pair_exp_mean")).otherwise(None))
# Ratio dernière valeur vs moyenne expanding
full = full.withColumn("pair_last_vs_mean",
    F.when(F.col("pair_exp_mean") > 0, F.col("lag_1") / F.col("pair_exp_mean")).otherwise(None))

# Agence
w_ag = Window.partitionBy("code_agence").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("agence_exp_mean",   F.avg("quantite").over(w_ag))
full = full.withColumn("agence_exp_median", F.percentile_approx("quantite", 0.5).over(w_ag))

# Article
w_art = Window.partitionBy("code_article").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("article_exp_mean",   F.avg("quantite").over(w_art))
full = full.withColumn("article_exp_median", F.percentile_approx("quantite", 0.5).over(w_art))

# Ratio paire vs agence et paire vs article
full = full.withColumn("pair_vs_agence",
    F.when(F.col("agence_exp_mean") > 0, F.col("pair_exp_mean") / F.col("agence_exp_mean")).otherwise(None))
full = full.withColumn("pair_vs_article",
    F.when(F.col("article_exp_mean") > 0, F.col("pair_exp_mean") / F.col("article_exp_mean")).otherwise(None))

print("✅ 14 stats expanding")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1g. Stats same-week historiques (4 features)
# MAGIC Signal saisonnier fort : quelle est la moyenne de cette semaine précise les années passées ?

# COMMAND ----------

# Moyenne et max de la même semaine N les années précédentes
w_same_week = Window.partitionBy("code_agence", "code_article", "num_sem").orderBy("annee").rowsBetween(Window.unboundedPreceding, -1)

full = full.withColumn("same_week_mean", F.avg("quantite").over(w_same_week))
full = full.withColumn("same_week_max",  F.max("quantite").over(w_same_week))
full = full.withColumn("same_week_min",  F.min("quantite").over(w_same_week))
full = full.withColumn("same_week_std",  F.stddev("quantite").over(w_same_week))

print("✅ 4 same-week stats")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1h. Saisonnalité (6 features)

# COMMAND ----------

full = full.withColumn("sin_sem",  F.sin(2 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("cos_sem",  F.cos(2 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("sin_sem2", F.sin(4 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("cos_sem2", F.cos(4 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("sin_sem4", F.sin(8 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("cos_sem4", F.cos(8 * np.pi * F.col("num_sem") / 52))

print("✅ 6 saisonnalité (3 harmoniques)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1i. Enrichissement agences (5 features)

# COMMAND ----------

agence_features = agences.select(
    "code_agence",
    F.col("region").alias("ag_region"),
    F.col("secteur").alias("ag_secteur"),
    F.col("departement").alias("ag_departement"),
    F.col("latitude").alias("ag_latitude"),
    F.col("longitude").alias("ag_longitude"),
)
full = full.join(agence_features, "code_agence", "left")

print("✅ 5 features agences")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1j. Enrichissement articles (7 features)

# COMMAND ----------

article_features = articles.select(
    "code_agence", "code_article",
    F.col("famille").alias("art_famille"),
    F.col("sous_famille").alias("art_sous_famille"),
    F.col("marque").alias("art_marque"),
    F.col("specialite").alias("art_specialite"),
    F.col("unite_vente").alias("art_unite_vente"),
    F.col("article_mdd").alias("art_mdd"),
    F.col("Gamme").alias("art_gamme"),
)
full = full.join(article_features, ["code_agence", "code_article"], "left")

print("✅ 7 features articles")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1k. Enrichissement facturation (12 features)

# COMMAND ----------

factu_agg = (
    factu.groupBy("code_agence", "code_article")
    .agg(
        F.round(F.avg("sum_quantite"), 2).alias("fac_avg_qty"),
        F.round(F.avg("nb_achats"), 2).alias("fac_avg_nb_achats"),
        F.round(F.sum("sum_montant") / (F.sum("sum_quantite") + 0.001), 2).alias("fac_prix_unit"),
        F.round(F.avg("sum_montant"), 2).alias("fac_avg_montant"),
        F.round(F.sum("nb_achats_par_professionnels") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_pro"),
        F.round(F.sum("nb_achats_par_particuliers") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_part"),
        F.round(F.sum("nb_ventes_magasins") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_magasin"),
        F.round(F.avg("nb_chantiers"), 2).alias("fac_avg_chantiers"),
        F.round(F.avg("max_quantite"), 2).alias("fac_avg_max_qty"),
        F.round(F.avg("min_quantite"), 2).alias("fac_avg_min_qty"),
        F.round(F.stddev("sum_quantite"), 2).alias("fac_std_qty"),
        F.round(F.avg("max_montant"), 2).alias("fac_avg_max_montant"),
    )
)
full = full.join(factu_agg, ["code_agence", "code_article"], "left")

print("✅ 12 features facturation")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1l. Lag 52 corrigé tendance (2 features)

# COMMAND ----------

full = full.withColumn("lag_52_trend_corr",
    F.when((F.col("lag_52").isNotNull()) & (F.col("trend_4_52").isNotNull()),
           F.col("lag_52") * F.col("trend_4_52")
    ).otherwise(F.col("lag_52"))
)

full = full.withColumn("same_week_trend_corr",
    F.when((F.col("same_week_mean").isNotNull()) & (F.col("trend_4_52").isNotNull()),
           F.col("same_week_mean") * F.col("trend_4_52")
    ).otherwise(F.col("same_week_mean"))
)

print("✅ 2 lag corrigés tendance")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1m. Interactions (6 features)

# COMMAND ----------

# Saisonnalité × niveau de la paire
full = full.withColumn("pair_mean_x_sin", F.col("pair_exp_mean") * F.col("sin_sem"))
full = full.withColumn("pair_mean_x_cos", F.col("pair_exp_mean") * F.col("cos_sem"))

# Volume paire × tendance
full = full.withColumn("pair_mean_x_trend", F.col("pair_exp_mean") * F.col("trend_4_52"))

# Prix × quantité
full = full.withColumn("fac_prix_x_pairmean",
    F.when(F.col("fac_prix_unit").isNotNull(), F.col("fac_prix_unit") * F.col("pair_exp_mean")).otherwise(None))

# Article MDD × zero_rate
full = full.withColumn("mdd_x_zero_rate",
    F.when(F.col("art_mdd").isNotNull(), F.col("art_mdd") * F.col("zero_rate_52")).otherwise(None))

# Latitude × saisonnalité (effet géo-saisonnier)
full = full.withColumn("lat_x_sin", F.col("ag_latitude") * F.col("sin_sem"))

print("✅ 6 interactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Conversion Pandas

# COMMAND ----------

cat_cols = ["ag_region", "ag_secteur", "art_famille", "art_sous_famille",
            "art_marque", "art_specialite", "art_unite_vente", "art_gamme"]
 
# Ne construire les indexers que pour les colonnes présentes dans le DataFrame
existing_cat = [c for c in cat_cols if c in full.columns]
missing_cat  = [c for c in cat_cols if c not in full.columns]
if missing_cat:
    print(f"⚠️  Colonnes absentes ignorées : {missing_cat}")
 
indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in existing_cat
]
 
if indexers:
    pipe      = Pipeline(stages=indexers)
    full      = pipe.fit(full).transform(full)
    # Remplacer les colonnes string par leurs versions indexées
    for c in existing_cat:
        full = full.drop(c).withColumnRenamed(c + "_idx", c)
 
print(f"✅ {len(existing_cat)} colonnes catégorielles encodées dans Spark")

# COMMAND ----------

# Identifiants nécessaires pour les masks + soumission
id_cols   = ["semaine", "code_agence", "code_article", "quantite", "annee", "sem_idx", "is_test"]
 
# Toutes les features générées (tout sauf les id_cols)
id_set       = set(id_cols)
feature_cols = [c for c in full.columns if c not in id_set]
 
cols_to_collect = id_cols + feature_cols
# Sécurité : ne garder que les colonnes qui existent réellement
cols_to_collect = [c for c in cols_to_collect if c in full.columns]
 
print(f"📦 Colonnes à collecter : {len(cols_to_collect)}  "
      f"(dont {len(feature_cols)} features + {len(id_cols)} ids)")

# COMMAND ----------

#full.select(cols_to_collect).cache()
print("⏳ toPandas() en cours — uniquement les colonnes utiles...")
 
df = full.select(cols_to_collect).toPandas()
print(f"✅ Shape : {df.shape}  |  Mémoire : {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
 
# Les colonnes catégorielles sont déjà numériques (float issues de StringIndexer)
# On les repasse en int puis en category pour LightGBM
for c in existing_cat:
    if c in df.columns:
        df[c] = df[c].fillna(-1).astype(int).astype("category")
 
print("✅ Catégorielles converties en dtype category")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature list complète

# COMMAND ----------

exclude = {"semaine", "code_agence", "code_article", "quantite", "annee", "sem_idx", "is_test"}
ALL_FEATURES = [c for c in df.columns if c not in exclude]

TARGET = "quantite"

# Split
train_mask = (df["is_test"] == 0) & (df["semaine"] < "2025-01")
val_mask   = (df["is_test"] == 0) & (df["semaine"].between("2025-01", "2025-26"))
test_mask  = df["is_test"] == 1

print(f"Total features générées : {len(ALL_FEATURES)}")
print(f"Train: {train_mask.sum():,}  |  Val: {val_mask.sum():,}  |  Test: {test_mask.sum():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. FEATURE SELECTION

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4a. Fonctions utilitaires

# COMMAND ----------

def wape_score(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + 1e-10)

def wape_eval(y_pred, dataset):
    y_true = dataset.get_label()
    wape = np.sum(np.abs(y_pred - y_true)) / (np.sum(y_true) + 1e-10)
    return "wape", wape, False

PARAMS = {
    "objective":              "tweedie",
    "tweedie_variance_power": 1.5,
    "metric":                 "None",
    "learning_rate":          0.05,
    "num_leaves":             127,
    "min_child_samples":      20,
    "feature_fraction":       0.8,
    "bagging_fraction":       0.8,
    "bagging_freq":           1,
    "reg_alpha":              0.1,
    "reg_lambda":             1.0,
    "n_jobs":                 -1,
    "seed":                   42,
    "verbose":                -1,
}

def evaluate_features(features, df=df, train_mask=train_mask, val_mask=val_mask,
                       params=PARAMS, cat_cols=cat_cols):
    """Entraîne un LightGBM et retourne le WAPE validation."""
    cat_f = [c for c in cat_cols if c in features]
    
    X_tr = df.loc[train_mask, features]
    y_tr = df.loc[train_mask, TARGET]
    X_va = df.loc[val_mask, features]
    y_va = df.loc[val_mask, TARGET]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_f, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_f, reference=dtrain, free_raw_data=False)
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        valid_names=["val"],
        feval=wape_eval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, min_delta=1e-4),
            lgb.log_evaluation(period=0),  # silent
        ],
    )
    
    y_pred = model.predict(X_va, num_iteration=model.best_iteration)
    y_pred = np.clip(y_pred, 0, None)
    score = wape_score(y_va.values, y_pred)
    
    importance = dict(zip(features, model.feature_importance(importance_type="gain")))
    
    return {
        "wape": score,
        "best_iter": model.best_iteration,
        "importance": importance,
        "model": model,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4b. Nettoyage colonnes triviales

# COMMAND ----------

# Supprimer constantes, quasi-constantes, full-null
clean_features = []
dropped_trivial = []

for f in ALL_FEATURES:
    col = df.loc[train_mask, f]
    n_unique = col.nunique(dropna=False)
    pct_null = col.isna().mean()
    
    if n_unique <= 1:
        dropped_trivial.append((f, "constante"))
    elif pct_null > 0.95:
        dropped_trivial.append((f, f"null {pct_null*100:.0f}%"))
    else:
        clean_features.append(f)

print(f"Features après nettoyage : {len(clean_features)} / {len(ALL_FEATURES)}")
if dropped_trivial:
    print(f"Supprimées ({len(dropped_trivial)}) :")
    for f, reason in dropped_trivial:
        print(f"  ✗ {f} ({reason})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4c. Baseline score (toutes les features nettoyées)

# COMMAND ----------

print(f"⏳ Évaluation baseline avec {len(clean_features)} features...")
baseline = evaluate_features(clean_features)
baseline_wape = baseline["wape"]
print(f"\n{'='*60}")
print(f"  BASELINE WAPE : {baseline_wape:.5f}  ({len(clean_features)} features, {baseline['best_iter']} iters)")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4d. Table d'importance

# COMMAND ----------

imp_df = pd.DataFrame([
    {"feature": f, "gain": baseline["importance"].get(f, 0)}
    for f in clean_features
]).sort_values("gain", ascending=True)

# Features à gain nul
zero_gain = imp_df[imp_df["gain"] == 0]["feature"].tolist()
low_gain  = imp_df[(imp_df["gain"] > 0) & (imp_df["gain"] < imp_df["gain"].quantile(0.15))]["feature"].tolist()

print(f"Features gain=0       : {len(zero_gain)}")
print(f"Features gain<P15     : {len(low_gain)}")
print(f"\nTop 30 features :")
print(imp_df.sort_values("gain", ascending=False).head(30).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4e. Boucle de suppression

# COMMAND ----------

# Ordre de test : d'abord gain=0, puis faibles gains
candidates = zero_gain + low_gain
# Ajouter le reste trié par gain croissant (on teste les plus faibles en premier)
remaining = [f for f in imp_df["feature"].tolist() if f not in candidates]
candidates = candidates + remaining

selected = clean_features.copy()
current_wape = baseline_wape
history = []
tolerance = 0.001  # on accepte si WAPE ne monte pas de plus de 0.1%

print(f"⏳ Boucle de suppression sur {len(candidates)} features...")
print(f"   Tolérance : {tolerance*100:.1f}%")
print(f"   Score départ : {current_wape:.5f}\n")

n_accepted = 0
recalc_every = 7  # recalculer les importances tous les 7 retraits

for i, feature in enumerate(candidates):
    if feature not in selected:
        continue
    
    trial = [f for f in selected if f != feature]
    
    try:
        result = evaluate_features(trial)
        trial_wape = result["wape"]
    except Exception as e:
        print(f"  ⚠️ Erreur sur {feature}: {e}")
        continue
    
    accepted = trial_wape <= current_wape * (1 + tolerance)
    delta = trial_wape - current_wape
    
    history.append({
        "step": i,
        "feature": feature,
        "old_wape": current_wape,
        "new_wape": trial_wape,
        "delta": delta,
        "accepted": accepted,
        "n_features": len(trial) if accepted else len(selected),
    })
    
    status = "✅ DROP" if accepted else "❌ KEEP"
    print(f"  [{i+1:>3}/{len(candidates)}] {status} {feature:<30}  WAPE: {trial_wape:.5f} (Δ={delta:+.5f})")
    
    if accepted:
        selected = trial
        current_wape = trial_wape
        n_accepted += 1
        
        # Recalculer les importances périodiquement
        if n_accepted % recalc_every == 0:
            print(f"\n  🔄 Recalcul importance après {n_accepted} suppressions...")
            recalc = evaluate_features(selected)
            current_wape = recalc["wape"]
            # Réordonner les candidates restantes
            new_imp = recalc["importance"]
            still_to_test = [f for f in candidates[i+1:] if f in selected]
            still_to_test.sort(key=lambda f: new_imp.get(f, 0))
            candidates = candidates[:i+1] + still_to_test
            print(f"  🔄 WAPE recalculé : {current_wape:.5f}  |  Restant à tester : {len(still_to_test)}\n")

print(f"\n{'='*60}")
print(f"  RÉSULTAT SÉLECTION")
print(f"  Features départ  : {len(clean_features)}")
print(f"  Features finales : {len(selected)}")
print(f"  Supprimées       : {len(clean_features) - len(selected)}")
print(f"  WAPE départ      : {baseline_wape:.5f}")
print(f"  WAPE final       : {current_wape:.5f}")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4f. Test suppression par blocs

# COMMAND ----------

# Tester la suppression de groupes logiques
block_tests = {
    "lags_secondaires": [f for f in selected if f.startswith("lag_") and f not in ["lag_1", "lag_4", "lag_13", "lag_26", "lag_52"]],
    "rolling_min": [f for f in selected if f.startswith("roll_min_")],
    "rolling_skew": [f for f in selected if f.startswith("roll_skew_")],
    "rolling_cv": [f for f in selected if f.startswith("roll_cv_")],
    "harmoniques_hautes": [f for f in selected if f in ["sin_sem4", "cos_sem4"]],
    "interactions": [f for f in selected if f in ["pair_mean_x_sin", "pair_mean_x_cos", "pair_mean_x_trend", "fac_prix_x_pairmean", "mdd_x_zero_rate", "lat_x_sin"]],
}

print("⏳ Test suppression par blocs :\n")
for block_name, block_features in block_tests.items():
    block_in_selected = [f for f in block_features if f in selected]
    if len(block_in_selected) == 0:
        continue
    
    trial = [f for f in selected if f not in block_in_selected]
    result = evaluate_features(trial)
    trial_wape = result["wape"]
    delta = trial_wape - current_wape
    accepted = trial_wape <= current_wape * (1 + tolerance)
    
    status = "✅ DROP BLOC" if accepted else "❌ KEEP BLOC"
    print(f"  {status} {block_name:<25} ({len(block_in_selected)} features)  WAPE: {trial_wape:.5f} (Δ={delta:+.5f})")
    
    if accepted:
        selected = trial
        current_wape = trial_wape
        history.append({
            "step": f"bloc_{block_name}",
            "feature": block_name,
            "old_wape": current_wape - delta,
            "new_wape": trial_wape,
            "delta": delta,
            "accepted": True,
            "n_features": len(selected),
        })

print(f"\n  Features finales après blocs : {len(selected)}")
print(f"  WAPE final : {current_wape:.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Résultats de la sélection

# COMMAND ----------

# Features finales
print(f"\n{'='*60}")
print(f"  FEATURES SÉLECTIONNÉES ({len(selected)})")
print(f"{'='*60}")
for f in sorted(selected):
    print(f"  • {f}")

# Features supprimées
dropped = [f for f in clean_features if f not in selected]
print(f"\n{'='*60}")
print(f"  FEATURES SUPPRIMÉES ({len(dropped)})")
print(f"{'='*60}")
for f in sorted(dropped):
    print(f"  ✗ {f}")

# Historique
hist_df = pd.DataFrame(history)
print(f"\n{'='*60}")
print(f"  HISTORIQUE DE SÉLECTION")
print(f"{'='*60}")
print(hist_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Training final + Prédictions

# COMMAND ----------

FINAL_FEATURES = selected

# Train + Val → modèle final
print(f"⏳ Training final avec {len(FINAL_FEATURES)} features...")

cat_f = [c for c in cat_cols if c in FINAL_FEATURES]

X_train = df.loc[train_mask, FINAL_FEATURES]
y_train = df.loc[train_mask, TARGET]
X_val   = df.loc[val_mask, FINAL_FEATURES]
y_val   = df.loc[val_mask, TARGET]
X_test  = df.loc[test_mask, FINAL_FEATURES]

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_f, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_f, reference=dtrain, free_raw_data=False)

# Trouver best iteration
model_val = lgb.train(
    PARAMS, dtrain,
    num_boost_round=2000,
    valid_sets=[dval], valid_names=["val"],
    feval=wape_eval,
    callbacks=[lgb.early_stopping(50, min_delta=1e-4), lgb.log_evaluation(100)],
)

best_iter = model_val.best_iteration
y_val_pred = np.clip(model_val.predict(X_val, num_iteration=best_iter), 0, None)
final_val_wape = wape_score(y_val.values, y_val_pred)

print(f"\n✅ Best iter: {best_iter}  |  WAPE val: {final_val_wape:.5f}")

# Re-train sur tout
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])
dfull = lgb.Dataset(X_full, label=y_full, categorical_feature=cat_f, free_raw_data=False)
model_final = lgb.train(PARAMS, dfull, num_boost_round=best_iter)

# Prédictions
y_test_pred = np.clip(model_final.predict(X_test), 0, None)
y_test_pred = np.round(y_test_pred).astype(int)

print(f"✅ Prédictions : {len(y_test_pred):,} lignes  |  Mean: {y_test_pred.mean():.2f}  |  %0: {(y_test_pred==0).mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Blend optionnel

# COMMAND ----------

lag52_val = df.loc[val_mask, "lag_52"].fillna(0).values

best_alpha = 1.0
best_blend_wape = final_val_wape

for a in np.arange(0.0, 1.01, 0.05):
    b = a * y_val_pred + (1 - a) * lag52_val
    w = wape_score(y_val.values, np.round(np.clip(b, 0, None)).astype(int))
    if w < best_blend_wape:
        best_blend_wape = w
        best_alpha = a

print(f"Best alpha  : {best_alpha:.2f}")
print(f"WAPE modèle : {final_val_wape:.5f}")
print(f"WAPE blend  : {best_blend_wape:.5f}")

if best_blend_wape < final_val_wape:
    print(f"✅ Blend améliore ! alpha={best_alpha:.2f}")
    lag52_test = df.loc[test_mask, "lag_52"].fillna(0).values
    y_test_pred = np.round(np.clip(
        best_alpha * model_final.predict(X_test) + (1 - best_alpha) * lag52_test,
        0, None
    )).astype(int)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sauvegarde

# COMMAND ----------

submission = df.loc[test_mask, ["semaine", "code_agence", "code_article"]].copy()
submission["quantite"] = y_test_pred

n_expected = 272344
assert len(submission) == n_expected, f"⚠️ {len(submission)} != {n_expected}"

submission_sp = spark.createDataFrame(submission)
submission_sp = (
    submission_sp
    .withColumn("code_agence", F.col("code_agence").cast("long"))
    .withColumn("code_article", F.col("code_article").cast("long"))
    .withColumn("quantite", F.col("quantite").cast("long"))
)

table_name = f"workspace.default.predictions_equipe_{TEAM_NAME}"
submission_sp.write.mode("overwrite").saveAsTable(table_name)

print(f"\n✅ Sauvegardé : {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Sauvegarde résultats sélection

# COMMAND ----------

# Sauvegarder les listes pour réutilisation
results = {
    "selected_features": selected,
    "dropped_features": dropped,
    "dropped_trivial": dropped_trivial,
    "baseline_wape": baseline_wape,
    "final_wape": current_wape,
    "n_features_start": len(clean_features),
    "n_features_end": len(selected),
}

print(json.dumps(results, indent=2, default=str))
