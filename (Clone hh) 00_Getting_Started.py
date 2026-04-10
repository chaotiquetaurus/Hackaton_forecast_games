# Databricks notebook source
# MAGIC %md
# MAGIC # Hackathon SGDB France 2026 — Guide de démarrage
# MAGIC
# MAGIC **Databricks × CentraleSupelec × MARGO**
# MAGIC
# MAGIC > 💡 **Pour commencer :** faites une copie de ce notebook dans votre espace personnel (*File → Clone*), puis travaillez depuis votre copie.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Objectif
# MAGIC
# MAGIC Prédire les **quantités de ventes hebdomadaires** pour chaque combinaison **agence × article** sur le second semestre 2025 (semaines 2025-27 à 2025-52).
# MAGIC
# MAGIC Votre score est calculé avec le **WAPE** (Weighted Absolute Percentage Error) :
# MAGIC
# MAGIC $$\text{WAPE} = \frac{\sum |\hat{y} - y|}{\sum y}$$
# MAGIC
# MAGIC **Plus le WAPE est bas, mieux c'est.** Un WAPE de 0 = prédiction parfaite.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Plan de ce notebook
# MAGIC
# MAGIC | Étape | Description |
# MAGIC |-------|-------------|
# MAGIC | **1** | Configurer votre équipe |
# MAGIC | **2** | Explorer les données |
# MAGIC | **3** | Construire une baseline |
# MAGIC | **4** | Auto-évaluer votre WAPE |
# MAGIC | **5** | Sauvegarder vos prédictions |
# MAGIC | **6** | Soumettre via l'application |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Étape 1 — Configurer votre équipe
# MAGIC
# MAGIC **Commencez par définir le nom de votre équipe ici.** Ce nom sera utilisé dans tout le notebook.
# MAGIC
# MAGIC > Utilisez uniquement des lettres minuscules, chiffres et underscores.
# MAGIC > Exemple : `data_pioneers`, `team_42`, `neural_nets`

# COMMAND ----------

# ⚠️  MODIFIEZ CE NOM — un seul endroit à changer pour tout le notebook
NOM_EQUIPE = "Telecaca"   # ← remplacez par le nom de votre équipe
#PARTE DE MON CLONE BANDE DE VOLEUR
# Ne touchez pas au reste
TABLE_PREDICTIONS = f"workspace.default.predictions_equipe_{NOM_EQUIPE}"
print(f"Votre table de prédictions : {TABLE_PREDICTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Étape 2 — Explorer les données
# MAGIC
# MAGIC Cinq tables sont disponibles dans le catalogue `workspace.default`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table d'entraînement — `histo_ventes_train`
# MAGIC
# MAGIC L'historique des ventes réelles. C'est votre principale source d'information.
# MAGIC
# MAGIC | Colonne | Type | Description |
# MAGIC |---------|------|-------------|
# MAGIC | `semaine` | STRING | Semaine ISO format `"YYYY-WW"` — ex : `"2024-15"` |
# MAGIC | `code_agence` | LONG | Identifiant de l'agence |
# MAGIC | `code_article` | LONG | Identifiant de l'article |
# MAGIC | `quantite` | LONG | Quantité vendue (peut être 0) |

# COMMAND ----------

display(spark.table("workspace.default.histo_ventes_train").limit(10))

# COMMAND ----------

# Statistiques générales sur le jeu d'entraînement
from pyspark.sql import functions as F

train_df = spark.table("workspace.default.histo_ventes_train")

print("=== Statistiques histo_ventes_train ===")
print(f"Nombre de lignes     : {train_df.count():,}")
print(f"Semaine min          : {train_df.agg(F.min('semaine')).collect()[0][0]}")
print(f"Semaine max          : {train_df.agg(F.max('semaine')).collect()[0][0]}")
print(f"Nombre d'agences     : {train_df.select('code_agence').distinct().count():,}")
print(f"Nombre d'articles    : {train_df.select('code_article').distinct().count():,}")

nb_total = train_df.count()
nb_zeros = train_df.filter(F.col("quantite") == 0).count()
print(f"Lignes avec quantité = 0 : {nb_zeros:,} ({100*nb_zeros/nb_total:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Table de test — `histo_ventes_test`
# MAGIC
# MAGIC Les lignes que **vous devez prédire**. Même structure que le train, mais `quantite` est vide.
# MAGIC
# MAGIC > Il y a **272 344 lignes** à prédire. Vous devez toutes les couvrir.
# MAGIC > Une ligne manquante est comptée comme une prédiction de **0**.

# COMMAND ----------

test_df = spark.table("workspace.default.histo_ventes_test")
display(test_df.limit(10))

# COMMAND ----------

print(f"Lignes à prédire : {test_df.count():,}")
print(f"Semaine min : {test_df.agg(F.min('semaine')).collect()[0][0]}")
print(f"Semaine max : {test_df.agg(F.max('semaine')).collect()[0][0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tables de référence
# MAGIC
# MAGIC Ces tables peuvent enrichir vos features.
# MAGIC
# MAGIC | Table | Contenu |
# MAGIC |-------|---------|
# MAGIC | `donnees_agence` | Caractéristiques des agences (région, taille, localisation…) |
# MAGIC | `donnees_articles` | Caractéristiques des articles (famille, catégorie, saisonnalité…) |
# MAGIC | `donnees_facturation` | Données de facturation (prix, montants, comportements d'achat…) |

# COMMAND ----------

display(spark.table("workspace.default.donnees_agence").limit(5))

# COMMAND ----------

display(spark.table("workspace.default.donnees_articles").limit(5))

# COMMAND ----------

display(spark.table("workspace.default.donnees_facturation").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Étape 3 — Construire une baseline
# MAGIC
# MAGIC Pour démarrer rapidement, voici une **baseline saisonnière N-1** : pour chaque ligne à prédire, on prend la quantité vendue la même semaine l'année précédente. Si aucune donnée N-1 n'existe, on prédit 0.
# MAGIC
# MAGIC C'est simple mais souvent difficile à battre sur des données saisonnières !
# MAGIC
# MAGIC > **Scores de référence à battre :**
# MAGIC > - Baseline saisonnière simple → WAPE ≈ 1.387
# MAGIC > - Blend pondéré N-1 + moyenne → WAPE ≈ 1.259
# MAGIC
# MAGIC Regardez aussi les notebooks **`01_LightGBM_Approche1`** et **`02_XGBoost_Approche2`** pour des approches ML plus avancées.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import LongType

train_df = spark.table("workspace.default.histo_ventes_train")
test_df  = spark.table("workspace.default.histo_ventes_test")

# Construire la clé de jointure N-1 : "2025-27" → "2024-27"
def semaine_n_moins_1(col):
    annee   = F.split(col, "-")[0].cast("int")
    num_sem = F.split(col, "-")[1]
    return F.concat((annee - 1).cast("string"), F.lit("-"), num_sem)

# Historique N-1 : renommer quantite → quantite_n1 et semaine → semaine_n1
train_n1 = (
    train_df
    .select(
        F.col("semaine").alias("semaine_n1"),
        "code_agence",
        "code_article",
        F.col("quantite").alias("quantite_n1"),
    )
)

# Jointure : pour chaque ligne de test, trouver la quantité N-1
predictions_df = (
    test_df
    .withColumn("semaine_n1", semaine_n_moins_1(F.col("semaine")))
    .join(train_n1, on=["semaine_n1", "code_agence", "code_article"], how="left")
    .withColumn("quantite", F.coalesce(F.col("quantite_n1"), F.lit(0)).cast(LongType()))
    .select("semaine", "code_agence", "code_article", "quantite")
)

print(f"Nombre de prédictions générées : {predictions_df.count():,}")
display(predictions_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Étape 4 — Auto-évaluer votre WAPE
# MAGIC
# MAGIC Les semaines **2025-01 à 2025-26** sont dans `histo_ventes_train` — vous connaissez les vraies valeurs.
# MAGIC Utilisez-les comme **jeu de validation** pour estimer votre WAPE avant de soumettre officiellement.
# MAGIC
# MAGIC **Recommandation :** entraînez votre modèle sur les données **avant 2025-01**, validez sur **2025-01 à 2025-26**.

# COMMAND ----------

def compute_wape(predictions_df, actuals_df, pred_col="quantite", actual_col="quantite_actuel"):
    """
    Calcule le WAPE entre un DataFrame de prédictions et les vraies valeurs.
    Les deux DataFrames doivent avoir : semaine, code_agence, code_article.
    """
    joined = predictions_df.join(
        actuals_df,
        on=["semaine", "code_agence", "code_article"],
        how="inner",
    )
    row = joined.agg(
        (
            F.sum(F.abs(F.col(pred_col).cast("double") - F.col(actual_col).cast("double")))
            / (F.sum(F.col(actual_col).cast("double")) + F.lit(1e-10))
        ).alias("wape")
    ).collect()[0]
    return float(row["wape"])


# Exemple : calcul du WAPE de validation pour la baseline N-1
# (on génère des prédictions sur les semaines 2025-01 à 2025-26)

train_avant_2025 = train_df.filter(F.col("semaine") < "2025-01")
val_actuel = (
    train_df
    .filter(F.col("semaine").between("2025-01", "2025-26"))
    .select("semaine", "code_agence", "code_article", F.col("quantite").alias("quantite_actuel"))
)

# Construire les prédictions N-1 sur la période de validation
train_n1_val = (
    train_avant_2025
    .select(
        F.col("semaine").alias("semaine_n1"),
        "code_agence",
        "code_article",
        F.col("quantite").alias("quantite_n1"),
    )
)

# Pour chaque ligne de validation, chercher la quantité de la même semaine l'année d'avant
val_preds = (
    val_actuel
    .withColumn("semaine_n1", semaine_n_moins_1(F.col("semaine")))
    .join(train_n1_val, on=["semaine_n1", "code_agence", "code_article"], how="left")
    .withColumn("quantite", F.coalesce(F.col("quantite_n1"), F.lit(0)).cast(LongType()))
    .select("semaine", "code_agence", "code_article", "quantite")
)

wape_val = compute_wape(val_preds, val_actuel)
print(f"WAPE de validation (baseline N-1) : {wape_val:.4f}")
print()
print("Scores de référence :")
print("  Baseline saisonnière  → WAPE ≈ 1.3870")
print("  Blend pondéré         → WAPE ≈ 1.2590")
print()
if wape_val < 1.259:
    print("Vous battez la référence ! Pensez à soumettre officiellement.")
else:
    print("Continuez à améliorer votre modèle avant de soumettre !")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Étape 5 — Sauvegarder vos prédictions
# MAGIC
# MAGIC Le format attendu est simple : **4 colonnes**, une ligne par combinaison à prédire.
# MAGIC
# MAGIC | Colonne | Type | Description |
# MAGIC |---------|------|-------------|
# MAGIC | `semaine` | STRING | Format `"YYYY-WW"` — ex : `"2025-27"` |
# MAGIC | `code_agence` | LONG | Identifiant de l'agence |
# MAGIC | `code_article` | LONG | Identifiant de l'article |
# MAGIC | `quantite` | LONG | Votre prédiction (≥ 0) |
# MAGIC
# MAGIC > **Couvrez toutes les 272 344 lignes de `histo_ventes_test`.**
# MAGIC > Une ligne manquante compte comme une prédiction de **0**.

# COMMAND ----------

# Vérifier que vos prédictions ont le bon format
print(f"Nombre de lignes dans vos prédictions : {predictions_df.count():,}")
print(f"Nombre attendu                         : 272 344")
print()
predictions_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sauvegarder dans votre table Delta
# MAGIC
# MAGIC Exécutez la cellule ci-dessous quand vos prédictions sont prêtes.
# MAGIC
# MAGIC > Vous pouvez écraser votre table autant de fois que vous voulez — **chaque soumission officielle est comptabilisée séparément** (5 max) depuis l'application.

# COMMAND ----------

# Décommentez et exécutez quand vos prédictions sont prêtes
predictions_df.write.mode("overwrite").saveAsTable(TABLE_PREDICTIONS)
print(f"✅ Prédictions sauvegardées dans : {TABLE_PREDICTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Si vous êtes plusieurs dans l'équipe
# MAGIC
# MAGIC La soumission est protégée : **seule la personne qui a créé la table peut soumettre** par défaut (Unity Catalog n'accorde l'accès MODIFY qu'au créateur de la table).
# MAGIC
# MAGIC Si vous voulez qu'un coéquipier puisse également soumettre, le créateur de la table doit lui donner accès :
# MAGIC
# MAGIC ```sql
# MAGIC GRANT MODIFY ON TABLE workspace.default.predictions_equipe_<NOM_EQUIPE>
# MAGIC TO 'email.coequipier@centrale-supelec.fr';
# MAGIC ```
# MAGIC
# MAGIC Cela garantit aussi que **personne d'une autre équipe ne peut soumettre à votre place** et gaspiller vos 5 soumissions.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Étape 6 — Soumettre via l'application
# MAGIC
# MAGIC Une fois votre table sauvegardée, rendez-vous sur l'application du hackathon :
# MAGIC
# MAGIC **[https://hackathon-sgdb-leaderboard-7474650176048310.aws.databricksapps.com](https://hackathon-sgdb-leaderboard-7474650176048310.aws.databricksapps.com)**
# MAGIC
# MAGIC ### Étapes de soumission
# MAGIC
# MAGIC 1. Cliquez sur l'onglet **"📤 Soumettre"**
# MAGIC 2. Tapez le nom de votre équipe dans le champ *(sans le préfixe `equipe_`)*
# MAGIC    — ex : tapez `mon_equipe`, pas `equipe_mon_equipe`
# MAGIC 3. Cliquez **"Calculer mon WAPE"** — votre score est calculé en direct
# MAGIC 4. Vérifiez le score affiché, puis cliquez **"🏆 Soumettre officiellement"**
# MAGIC 5. Votre score apparaît sur le leaderboard en temps réel
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC > ⚠️ **Limite : 5 soumissions par équipe.** Le score le plus récent est affiché sur le leaderboard.
# MAGIC > Utilisez "Calculer mon WAPE" autant que vous voulez pour vérifier votre score — cela ne consomme pas de soumission.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Conseils clés
# MAGIC
# MAGIC ### Données
# MAGIC - **~68 % de zéros** : prédire 0 pour les combinaisons agence × article rares est souvent optimal.
# MAGIC - **Saisonnalité forte** : la même semaine de l'année précédente est un signal puissant.
# MAGIC - **Enrichissez vos features** avec `donnees_agence` (région), `donnees_articles` (catégorie), `donnees_facturation` (prix).
# MAGIC
# MAGIC ### Feature engineering — par où commencer
# MAGIC
# MAGIC Les features les plus utiles sur ce type de données :
# MAGIC
# MAGIC | Catégorie | Exemples |
# MAGIC |-----------|----------|
# MAGIC | **Lags temporels** | Quantité à N-1, N-2, N-4, N-8, N-26, N-52 semaines |
# MAGIC | **Même semaine N-1** | Quantité de la même semaine l'année précédente — signal saisonnier fort |
# MAGIC | **Moyennes mobiles** | Moyenne sur 4, 12, 26 semaines glissantes (avec shift pour éviter la fuite) |
# MAGIC | **Taux de zéros** | Part des semaines à zéro sur les 4, 12, 52 dernières semaines |
# MAGIC | **Stats agence** | Moyenne, médiane, écart-type des ventes par agence |
# MAGIC | **Stats article** | Moyenne, médiane, taux de zéros par article |
# MAGIC | **Stats paire agence × article** | Moyenne, max, count par paire — très prédictif |
# MAGIC | **Saisonnalité** | `sin(2π × semaine / 52)` et `cos(2π × semaine / 52)` |
# MAGIC
# MAGIC > ⚠️ Attention à la **fuite de données** : les features basées sur l'historique doivent utiliser uniquement les données *antérieures* à la semaine prédite (utilisez `.shift(1)` avant les rolling).
# MAGIC
# MAGIC ### Stratégie
# MAGIC - Validez toujours sur **2025-01 à 2025-26** avant de soumettre officiellement
# MAGIC - Un blend de plusieurs modèles bat généralement un modèle seul
# MAGIC - Consultez le notebook **`LightGBM_Example`** pour un pipeline complet : feature engineering, entraînement LightGBM, et génération des prédictions
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Bonne chance ! 🏆**

# COMMAND ----------


