# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Pipeline Complet — Features enrichies + Tweedie WAPE
# MAGIC
# MAGIC Features complètes (lags, rolling, enrichissement, facturation, catégorielles)  
# MAGIC Training : Tweedie + custom WAPE eval + early stopping (ta logique)

# COMMAND ----------

!pip install lightgbm

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import lightgbm as lgb
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
print(f"Full: {full.count():,} lignes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering

# COMMAND ----------

# ---------------------------------------------------------------
# 2a. LAGS
# ---------------------------------------------------------------
w = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx")

for lag_n in [1, 2, 4, 8, 13, 26, 52]:
    full = full.withColumn(f"lag_{lag_n}", F.lag("quantite", lag_n).over(w))

# Lag 52 spécial : quantité brute même semaine N-1 (pour blend plus tard)
full = full.withColumn("lag_52_qty", F.lag("quantite", 52).over(w))

print("✅ Lags")

# COMMAND ----------

# ---------------------------------------------------------------
# 2b. ROLLING STATS (fenêtre AVANT = pas de leakage)
# ---------------------------------------------------------------
for win_size in [4, 8, 13, 26, 52]:
    w_roll = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(-win_size, -1)
    full = full.withColumn(f"roll_mean_{win_size}", F.avg("quantite").over(w_roll))
    full = full.withColumn(f"roll_std_{win_size}", F.stddev("quantite").over(w_roll))
    full = full.withColumn(f"roll_max_{win_size}", F.max("quantite").over(w_roll))

print("✅ Rolling stats")

# COMMAND ----------

# ---------------------------------------------------------------
# 2c. TAUX DE ZEROS
# ---------------------------------------------------------------
for win_size in [4, 13, 26, 52]:
    w_roll = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(-win_size, -1)
    full = full.withColumn(
        f"zero_rate_{win_size}",
        F.avg(F.when(F.col("quantite") == 0, 1.0).otherwise(0.0)).over(w_roll)
    )

print("✅ Zero rates")

# COMMAND ----------

# ---------------------------------------------------------------
# 2d. STATS EXPANDING (paire / agence / article)
# ---------------------------------------------------------------
w_exp_pair = Window.partitionBy("code_agence", "code_article").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("pair_mean", F.avg("quantite").over(w_exp_pair))
full = full.withColumn("pair_median", F.percentile_approx("quantite", 0.5).over(w_exp_pair))
full = full.withColumn("pair_max", F.max("quantite").over(w_exp_pair))
full = full.withColumn("pair_std", F.stddev("quantite").over(w_exp_pair))
full = full.withColumn("pair_count", F.count("quantite").over(w_exp_pair))
full = full.withColumn("pair_zero_rate", F.avg(F.when(F.col("quantite") == 0, 1.0).otherwise(0.0)).over(w_exp_pair))

w_exp_ag = Window.partitionBy("code_agence").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("agence_mean", F.avg("quantite").over(w_exp_ag))
full = full.withColumn("agence_median", F.percentile_approx("quantite", 0.5).over(w_exp_ag))

w_exp_art = Window.partitionBy("code_article").orderBy("sem_idx").rowsBetween(Window.unboundedPreceding, -1)
full = full.withColumn("article_mean", F.avg("quantite").over(w_exp_art))
full = full.withColumn("article_median", F.percentile_approx("quantite", 0.5).over(w_exp_art))

print("✅ Expanding stats")

# COMMAND ----------

# ---------------------------------------------------------------
# 2e. TENDANCES
# ---------------------------------------------------------------
# Court vs long terme
full = full.withColumn(
    "trend_4_13",
    F.when(F.col("roll_mean_13") > 0, F.col("roll_mean_4") / F.col("roll_mean_13")).otherwise(None)
)
full = full.withColumn(
    "trend_4_52",
    F.when(F.col("roll_mean_52") > 0, F.col("roll_mean_4") / F.col("roll_mean_52")).otherwise(None)
)
full = full.withColumn(
    "trend_13_52",
    F.when(F.col("roll_mean_52") > 0, F.col("roll_mean_13") / F.col("roll_mean_52")).otherwise(None)
)

# YoY ratio : lag_1 / lag_52 (le S-1 récent vs la même période l'an dernier)
full = full.withColumn(
    "yoy_ratio",
    F.when(F.col("lag_52") > 0, F.col("lag_1") / F.col("lag_52")).otherwise(None)
)

# Lag 52 corrigé par la tendance récente
# = ce qu'on aurait vendu la même semaine N-1 si on applique la tendance actuelle
full = full.withColumn(
    "lag_52_trend_corrected",
    F.when(
        (F.col("lag_52").isNotNull()) & (F.col("trend_4_52").isNotNull()),
        F.col("lag_52") * F.col("trend_4_52")
    ).otherwise(F.col("lag_52"))
)

print("✅ Tendances")

# COMMAND ----------

# ---------------------------------------------------------------
# 2f. SAISONNALITE
# ---------------------------------------------------------------
full = full.withColumn("sin_sem", F.sin(2 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("cos_sem", F.cos(2 * np.pi * F.col("num_sem") / 52))
# Harmonique 2 (capte les bi-annuels : creux été + creux Noël)
full = full.withColumn("sin_sem2", F.sin(4 * np.pi * F.col("num_sem") / 52))
full = full.withColumn("cos_sem2", F.cos(4 * np.pi * F.col("num_sem") / 52))

print("✅ Saisonnalité")

# COMMAND ----------

# ---------------------------------------------------------------
# 2g. ENRICHISSEMENT AGENCES
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

print("✅ Enrichissement agences")

# COMMAND ----------

# ---------------------------------------------------------------
# 2h. ENRICHISSEMENT ARTICLES (indexé par agence × article)
# ---------------------------------------------------------------
article_features = articles.select(
    "code_agence",
    "code_article",
    F.col("famille").alias("art_famille"),
    F.col("sous_famille").alias("art_sous_famille"),
    F.col("marque").alias("art_marque"),
    F.col("specialite").alias("art_specialite"),
    F.col("unite_vente").alias("art_unite_vente"),
    F.col("article_mdd").alias("art_mdd"),
    F.col("Gamme").alias("art_gamme"),
)
full = full.join(article_features, ["code_agence", "code_article"], "left")

print("✅ Enrichissement articles")

# COMMAND ----------

# ---------------------------------------------------------------
# 2i. ENRICHISSEMENT FACTURATION
# ---------------------------------------------------------------
factu_agg = (
    factu.groupBy("code_agence", "code_article")
    .agg(
        F.round(F.avg("sum_quantite"), 2).alias("fac_avg_qty"),
        F.round(F.avg("nb_achats"), 2).alias("fac_avg_nb_achats"),
        F.round(F.sum("sum_montant") / (F.sum("sum_quantite") + 0.001), 2).alias("fac_prix_unit"),
        F.round(F.sum("nb_achats_par_professionnels") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_pro"),
        F.round(F.sum("nb_achats_par_particuliers") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_part"),
        F.round(F.sum("nb_ventes_magasins") / (F.sum("nb_achats") + 1), 3).alias("fac_pct_magasin"),
        F.round(F.avg("nb_chantiers"), 2).alias("fac_avg_chantiers"),
        F.round(F.avg("sum_montant"), 2).alias("fac_avg_montant"),
    )
)
full = full.join(factu_agg, ["code_agence", "code_article"], "left")

print("✅ Enrichissement facturation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Conversion Pandas + Encoding catégorielles

# COMMAND ----------

df = full.toPandas()
print(f"Shape: {df.shape}")

# Colonnes catégorielles — LightGBM les gère nativement
cat_cols = ["ag_region", "ag_secteur", "art_famille", "art_sous_famille",
            "art_marque", "art_specialite", "art_unite_vente", "art_gamme"]

for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype("category")

print(f"✅ {len(cat_cols)} colonnes catégorielles encodées")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Définition des features + Split

# COMMAND ----------

FEATURES = [
    # --- Lags ---
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52", "lag_52_qty",
    
    # --- Rolling stats ---
    "roll_mean_4", "roll_mean_8", "roll_mean_13", "roll_mean_26", "roll_mean_52",
    "roll_std_4", "roll_std_8", "roll_std_13", "roll_std_26", "roll_std_52",
    "roll_max_4", "roll_max_8", "roll_max_13", "roll_max_26", "roll_max_52",
    
    # --- Zero rates ---
    "zero_rate_4", "zero_rate_13", "zero_rate_26", "zero_rate_52",
    
    # --- Pair / agence / article expanding ---
    "pair_mean", "pair_median", "pair_max", "pair_std", "pair_count", "pair_zero_rate",
    "agence_mean", "agence_median",
    "article_mean", "article_median",
    
    # --- Tendances ---
    "trend_4_13", "trend_4_52", "trend_13_52",
    "yoy_ratio", "lag_52_trend_corrected",
    
    # --- Temps + Saisonnalité ---
    "annee", "num_sem", "sin_sem", "cos_sem", "sin_sem2", "cos_sem2",
    
    # --- Enrichissement agences ---
    "ag_region", "ag_secteur", "ag_departement", "ag_latitude", "ag_longitude",
    
    # --- Enrichissement articles ---
    "art_famille", "art_sous_famille", "art_marque", "art_specialite",
    "art_unite_vente", "art_mdd", "art_gamme",
    
    # --- Facturation ---
    "fac_avg_qty", "fac_avg_nb_achats", "fac_prix_unit",
    "fac_pct_pro", "fac_pct_part", "fac_pct_magasin",
    "fac_avg_chantiers", "fac_avg_montant",
]

# Garder seulement les features qui existent dans le df
FEATURES = [f for f in FEATURES if f in df.columns]

TARGET = "quantite"

# Split
train_mask = (df["is_test"] == 0) & (df["semaine"] < "2025-01")
val_mask   = (df["is_test"] == 0) & (df["semaine"].between("2025-01", "2025-26"))
test_mask  = df["is_test"] == 1

X_train = df.loc[train_mask, FEATURES]
y_train = df.loc[train_mask, TARGET]
X_val   = df.loc[val_mask, FEATURES]
y_val   = df.loc[val_mask, TARGET]
X_test  = df.loc[test_mask, FEATURES]

print(f"Features: {len(FEATURES)}")
print(f"X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training — Ta logique (Tweedie + WAPE custom)

# COMMAND ----------

# WAPE custom metric
def wape_eval(y_pred, dataset):
    y_true = dataset.get_label()
    wape = np.sum(np.abs(y_pred - y_true)) / (np.sum(y_true) + 1e-10)
    return "wape", wape, False  # lower is better

# Identifier les colonnes catégorielles pour LightGBM
cat_features = [c for c in cat_cols if c in FEATURES]

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False)
dval   = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_features, reference=dtrain, free_raw_data=False)

params = {
    "objective":              "tweedie",
    "tweedie_variance_power": 1.5,
    "metric":                 "None",       # custom WAPE
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

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    valid_names=["train", "val"],
    feval=wape_eval,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, min_delta=1e-4),
        lgb.log_evaluation(period=50),
    ],
)

print(f"\n✅ Best iteration: {model.best_iteration}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Évaluation WAPE

# COMMAND ----------

def wape_score(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + 1e-10)

y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_val_pred = np.clip(y_val_pred, 0, None)
y_val_rounded = np.round(y_val_pred).astype(int)

val_wape = wape_score(y_val.values, y_val_rounded)
val_wape_raw = wape_score(y_val.values, y_val_pred)

print(f"{'='*60}")
print(f"  WAPE val (arrondi)   : {val_wape:.4f}")
print(f"  WAPE val (brut)      : {val_wape_raw:.4f}")
print(f"{'='*60}")
print(f"  Baseline saisonnière : ~1.387")
print(f"  Baseline blend       : ~1.259")
print(f"  Ton modèle précédent : ~1.100")
print(f"  Ce modèle            : {val_wape:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature Importance

# COMMAND ----------

importance = pd.DataFrame({
    "feature": FEATURES,
    "gain": model.feature_importance(importance_type="gain"),
    "split": model.feature_importance(importance_type="split"),
}).sort_values("gain", ascending=False)

print("🏆 Top 30 features (gain) :")
print(importance.head(30).to_string(index=False))

zero_imp = importance[importance["gain"] == 0]
if len(zero_imp) > 0:
    print(f"\n⚠️  {len(zero_imp)} features inutiles (gain=0) :")
    print(zero_imp["feature"].tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Re-train complet + Prédictions test

# COMMAND ----------

# Train sur tout (train + val)
X_full = pd.concat([X_train, X_val])
y_full = pd.concat([y_train, y_val])

dfull = lgb.Dataset(X_full, label=y_full, categorical_feature=cat_features, free_raw_data=False)

model_final = lgb.train(
    params,
    dfull,
    num_boost_round=model.best_iteration,
)

# Prédictions test
y_test_pred = model_final.predict(X_test)
y_test_pred = np.clip(y_test_pred, 0, None)
y_test_pred = np.round(y_test_pred).astype(int)

print(f"✅ Prédictions : {len(y_test_pred):,} lignes")
print(f"   Min: {y_test_pred.min()}  Max: {y_test_pred.max()}  Mean: {y_test_pred.mean():.2f}")
print(f"   % zéros: {(y_test_pred == 0).mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. (Optionnel) Blend avec baseline saisonnière

# COMMAND ----------

# La baseline N-1 est souvent complémentaire au ML
# Blend : alpha × model + (1-alpha) × lag_52
alpha = 0.7  # poids du modèle, à tuner sur val

lag52_val = df.loc[val_mask, "lag_52_qty"].fillna(0).values
blend_val = alpha * y_val_pred + (1 - alpha) * lag52_val
blend_val_rounded = np.round(np.clip(blend_val, 0, None)).astype(int)
blend_wape = wape_score(y_val.values, blend_val_rounded)

# Chercher le meilleur alpha
best_alpha = 0.7
best_blend_wape = blend_wape
for a in np.arange(0.0, 1.01, 0.05):
    b = a * y_val_pred + (1 - a) * lag52_val
    b_rounded = np.round(np.clip(b, 0, None)).astype(int)
    w = wape_score(y_val.values, b_rounded)
    if w < best_blend_wape:
        best_blend_wape = w
        best_alpha = a

print(f"Best blend alpha (modèle): {best_alpha:.2f}")
print(f"WAPE modèle seul  : {val_wape:.4f}")
print(f"WAPE blend optimal : {best_blend_wape:.4f}")

# Si le blend est meilleur, l'utiliser pour le test
if best_blend_wape < val_wape:
    print(f"\n✅ Le blend améliore ! Utilisation de alpha={best_alpha:.2f}")
    lag52_test = df.loc[test_mask, "lag_52_qty"].fillna(0).values
    y_test_pred = np.round(np.clip(
        best_alpha * model_final.predict(X_test) + (1 - best_alpha) * lag52_test,
        0, None
    )).astype(int)
else:
    print(f"\n→ Le modèle seul est meilleur, pas de blend")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Sauvegarde

# COMMAND ----------

submission = df.loc[test_mask, ["semaine", "code_agence", "code_article"]].copy()
submission["quantite"] = y_test_pred

n_expected = 272344
n_actual = len(submission)
print(f"Lignes : {n_actual:,} / {n_expected:,} attendues")
assert n_actual == n_expected, f"⚠️ {n_actual} != {n_expected}"

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
print(f"   → Allez soumettre sur l'app !")
