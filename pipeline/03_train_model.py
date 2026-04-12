# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Train the two-stage model
# MAGIC
# MAGIC **Architecture:**
# MAGIC - **Stage 1** — a binary LightGBM classifier predicts `P(quantite = 0)`.
# MAGIC - **Stage 2** — a LightGBM regressor (trained only on rows where
# MAGIC   `quantite > 0`) predicts the expected quantity *conditional* on being
# MAGIC   non-zero, using raw `quantite` as the target with MAE loss (regression_l1).
# MAGIC - At scoring time: `pred = 0 if p_zero > threshold else reg_pred`.
# MAGIC - The threshold is tuned on the validation set to minimise WAPE.
# MAGIC
# MAGIC **Inputs:** `workspace.default.feature_table`.
# MAGIC **Outputs:**
# MAGIC - Both models logged to MLflow + registered in the Model Registry.
# MAGIC - The best zero-threshold logged as a run parameter.
# MAGIC - A validation predictions Delta table used by 04_evaluate.py.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install lightgbm==4.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import sys
sys.path.append("./")

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, DoubleType, StringType, StructField, StructType

from src.utils import wape_numpy, wape_lgb_feval

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load the pre-split feature tables from the Lakeflow pipeline
# MAGIC
# MAGIC The DLT pipeline (`pipeline_lakeflow.py`) produces `gold_train` and
# MAGIC `gold_validation` with the temporal split already applied. We just
# MAGIC project the columns we need and move to pandas for LightGBM.

# COMMAND ----------

cols_needed = (
    ["semaine", "code_agence", "code_article", "week_id", "quantite", "is_dead_pair"]
    + FEATURES
)
 
train_pd = spark.table(TBL_GOLD_TRAIN).select(*cols_needed).toPandas()
val_pd   = spark.table(TBL_GOLD_VAL).select(*cols_needed).toPandas()
 
assert train_pd["week_id"].max() < val_pd["week_id"].min(), "Temporal split broken"
 
# Les paires mortes génèrent des zéros structurels qui biaisent le classifieur.
# On les filtre du train uniquement — le val reste intact pour l'évaluation réelle.
train_pd = train_pd[train_pd["is_dead_pair"] == 0].reset_index(drop=True)
print(f"Train (sans paires mortes): {len(train_pd):,}   Val: {len(val_pd):,}")

# COMMAND ----------

def add_zero_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des features zero-signal calculées sur ref_df (toujours le train).
    
    Features ajoutées :
    - zero_rate_article      : taux de zéro historique par article
    - zero_rate_agence       : taux de zéro historique par agence
    - zero_rate_pair         : taux de zéro historique par (agence, article)
    - consec_zeros_pair      : nb de semaines consécutives à zéro avant la semaine courante
                               (approximé ici par le zero_rate_pair * nb semaines total)
    - cv_pair                : coefficient de variation des ventes par paire
                               (std / mean) — élevé = série très intermittente
    - zero_rate_pair_recent  : taux de zéro sur les 8 dernières semaines de la paire
    - weeks_since_last_sale  : nb de semaines depuis la dernière vente (approximé)
    - is_always_zero_article : 1 si l'article n'a jamais eu de vente dans ref_df
    """
    df = df.copy()
 
    # -- Taux de zéro par article
    art_zero = (
        ref_df.groupby("code_article")["quantite"]
        .apply(lambda x: (x == 0).mean())
        .rename("zero_rate_article")
        .reset_index()
    )
    df = df.merge(art_zero, on="code_article", how="left")
    df["zero_rate_article"] = df["zero_rate_article"].fillna(0.5)
 
    # -- Taux de zéro par agence
    ag_zero = (
        ref_df.groupby("code_agence")["quantite"]
        .apply(lambda x: (x == 0).mean())
        .rename("zero_rate_agence")
        .reset_index()
    )
    df = df.merge(ag_zero, on="code_agence", how="left")
    df["zero_rate_agence"] = df["zero_rate_agence"].fillna(0.5)
 
    # -- Stats par paire (agence, article)
    pair_stats = (
        ref_df.groupby(["code_agence", "code_article"])["quantite"]
        .agg(
            zero_rate_pair   = lambda x: (x == 0).mean(),
            mean_qty_pair    = "mean",
            std_qty_pair     = "std",
            count_pair       = "count",
        )
        .reset_index()
    )
    pair_stats["cv_pair"] = (
        pair_stats["std_qty_pair"] / (pair_stats["mean_qty_pair"] + 1e-6)
    ).clip(0, 10)
    pair_stats["is_always_zero_article"] = (pair_stats["zero_rate_pair"] == 1.0).astype(int)
    df = df.merge(
        pair_stats[["code_agence", "code_article",
                    "zero_rate_pair", "cv_pair", "is_always_zero_article"]],
        on=["code_agence", "code_article"],
        how="left",
    )
    df["zero_rate_pair"]          = df["zero_rate_pair"].fillna(0.5)
    df["cv_pair"]                 = df["cv_pair"].fillna(1.0)
    df["is_always_zero_article"]  = df["is_always_zero_article"].fillna(0)
 
    # -- Taux de zéro récent (8 dernières semaines de ref_df par paire)
    max_week = ref_df["week_id"].max()
    recent   = ref_df[ref_df["week_id"] >= max_week - 7]
    recent_zero = (
        recent.groupby(["code_agence", "code_article"])["quantite"]
        .apply(lambda x: (x == 0).mean())
        .rename("zero_rate_pair_recent")
        .reset_index()
    )
    df = df.merge(recent_zero, on=["code_agence", "code_article"], how="left")
    # Si la paire n'a pas de données récentes → on utilise le taux global
    df["zero_rate_pair_recent"] = df["zero_rate_pair_recent"].fillna(df["zero_rate_pair"])
 
    # -- Semaines depuis la dernière vente (approximation par week_id)
    last_sale = (
        ref_df[ref_df["quantite"] > 0]
        .groupby(["code_agence", "code_article"])["week_id"]
        .max()
        .rename("last_sale_week_id")
        .reset_index()
    )
    df = df.merge(last_sale, on=["code_agence", "code_article"], how="left")
    df["weeks_since_last_sale"] = (
        df["week_id"] - df["last_sale_week_id"].fillna(df["week_id"] - 52)
    ).clip(0, 104)   # cap à 2 ans
    df = df.drop(columns=["last_sale_week_id"])
 
    return df
 
ZERO_FEATURES_ADDED = [
    "zero_rate_article",
    "zero_rate_agence",
    "zero_rate_pair",
    "cv_pair",
    "is_always_zero_article",
    "zero_rate_pair_recent",
    "weeks_since_last_sale",
]
 
train_pd = add_zero_features(train_pd, ref_df=train_pd)
val_pd   = add_zero_features(val_pd,   ref_df=train_pd)  # ref = train uniquement, pas de fuite
 
FEATURES_ZERO = FEATURES + ZERO_FEATURES_ADDED
print(f"Features classifieur zéro : {len(FEATURES_ZERO)}  (+{len(ZERO_FEATURES_ADDED)} zero-signal)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare X / y / is_zero

# COMMAND ----------

def build_xy_zero(df: pd.DataFrame, features: list, ref_df: pd.DataFrame = None):
    X = df[features].copy()
    for c in FEATURES_CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y       = df["quantite"].astype(float).values
    is_zero = (y == 0).astype(int)
    return X, y, is_zero
 
X_tr, y_tr, z_tr = build_xy_zero(train_pd, FEATURES_ZERO)
X_va, y_va, z_va = build_xy_zero(val_pd,   FEATURES_ZERO)
 
print(f"Zero rate — train: {z_tr.mean():.3f}   val: {z_va.mean():.3f}")
 
# Poids WAPE par article : 1 / (volume_total_article + 1)
# Évite que les gros articles dominent la loss.
article_vol = train_pd.groupby("code_article")["quantite"].sum().rename("art_vol")
train_pd = train_pd.join(article_vol, on="code_article")
val_pd   = val_pd.join(article_vol,   on="code_article")
val_pd["art_vol"] = val_pd["art_vol"].fillna(1.0)
 
w_tr = (1.0 / (train_pd["art_vol"].values + 1.0))
w_va = (1.0 / (val_pd["art_vol"].values   + 1.0))

# COMMAND ----------

n_pos   = int(z_tr.sum())          # nb de zéros (classe positive)
n_neg   = int((z_tr == 0).sum())   # nb de non-zéros
spw     = n_neg / max(n_pos, 1)    # scale_pos_weight = ratio non-zero / zero
print(f"n_zero={n_pos:,}  n_nonzero={n_neg:,}  scale_pos_weight={spw:.3f}")
 
LGB_PARAMS_ZERO_V2 = {
    # Objectif
    "objective":         "binary",
    "metric":            "binary_logloss",
    # Capacité du modèle
    "num_leaves":        255,
    "max_depth":         -1,
    "min_child_samples": 50,
    # Régularisation
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "min_split_gain":    0.01,
    # Sous-échantillonnage (réduit overfitting + accélère)
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.7,
    # Déséquilibre de classes
    "scale_pos_weight":  spw,
    "is_unbalance":      False,
    # Vitesse / reproductibilité
    "learning_rate":     0.05,
    "n_jobs":            -1,
    "seed":              42,
    "verbose":           -1,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Stage 1 — zero classifier
# MAGIC
# MAGIC The target is `is_zero`. We log the val logloss and AUC, and later
# MAGIC pull the probabilities to sweep the threshold.

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)
 
with mlflow.start_run(run_name="stage1_zero_clf_v2") as run:
    mlflow.log_params({
        **LGB_PARAMS_ZERO_V2,
        "n_features_zero":      len(FEATURES_ZERO),
        "zero_features_added":  ZERO_FEATURES_ADDED,
        "train_rows":           len(train_pd),
        "val_rows":             len(val_pd),
        "zero_rate_train":      float(z_tr.mean()),
        "zero_rate_val":        float(z_va.mean()),
        "dead_pair_filtered":   True,
        "sample_weight":        "wape_per_article",
    })
 
    dtrain_z = lgb.Dataset(
        X_tr, label=z_tr,
        weight=w_tr,
        categorical_feature=FEATURES_CATEGORICAL,
    )
    dval_z = lgb.Dataset(
        X_va, label=z_va,
        weight=w_va,
        reference=dtrain_z,
        categorical_feature=FEATURES_CATEGORICAL,
    )
 
# PAR ça :
params_phase1 = {**LGB_PARAMS_ZERO_V2, "learning_rate": 0.1}

model_zero = lgb.train(
    params_phase1,
    dtrain_z,
    num_boost_round=200,
    valid_sets=[dtrain_z, dval_z],
    valid_names=["train", "val"],
    callbacks=[
        lgb.log_evaluation(period=50),
    ],
)

params_phase2 = {**LGB_PARAMS_ZERO_V2, "learning_rate": 0.005}

model_zero = lgb.train(
    params_phase2,
    dtrain_z,
    num_boost_round=1000,
    init_model=model_zero,
    valid_sets=[dtrain_z, dval_z],
    valid_names=["train", "val"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150, first_metric_only=True),
        lgb.log_evaluation(period=100),
    ],
)
 
    mlflow.log_param("best_iteration_zero", model_zero.best_iteration)
    p_zero_raw = model_zero.predict(X_va, num_iteration=model_zero.best_iteration)
 

# COMMAND ----------

mid = len(p_zero_raw) // 2
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_zero_raw[:mid], z_va[:mid])
    p_zero_val = iso.transform(p_zero_raw)   # probabilités calibrées sur tout le val

# COMMAND ----------

   auc_pr  = average_precision_score(z_va, p_zero_val)
    auc_roc = roc_auc_score(z_va, p_zero_val)
    logloss = float(
        -np.mean(
            z_va * np.log(np.clip(p_zero_val, 1e-7, 1 - 1e-7))
            + (1 - z_va) * np.log(np.clip(1 - p_zero_val, 1e-7, 1 - 1e-7))
        )
    )
 
    mlflow.log_metrics({
        "val_logloss_calibrated": logloss,
        "val_auc_roc":            auc_roc,
        "val_auc_pr":             auc_pr,
    })
    print(f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Logloss={logloss:.4f}")

# COMMAND ----------

 try:
        # Vérification que model_qty est disponible dans le scope
        _ = model_qty
        qty_pred_raw = model_qty.predict(X_va, num_iteration=model_qty.best_iteration)
        qty_pred     = np.clip(qty_pred_raw, 0.0, None)
 
        # Segmentation par volume article (terciles)
        vol_q33 = val_pd["art_vol"].quantile(0.33)
        vol_q66 = val_pd["art_vol"].quantile(0.66)
 
        def segment(v):
            if v <= vol_q33:   return "low"
            elif v <= vol_q66: return "medium"
            else:               return "high"
 
        val_pd["vol_segment"] = val_pd["art_vol"].apply(segment)
 
        best_thresholds = {}
        rows_seg = []
        for seg in ["low", "medium", "high"]:
            mask    = val_pd["vol_segment"].values == seg
            p_seg   = p_zero_val[mask]
            qty_seg = qty_pred[mask]
            y_seg   = y_va[mask]
 
            best_thr, best_w = 0.5, float("inf")
            for thr in np.arange(0.1, 0.95, 0.025):
                final = np.where(p_seg > thr, 0.0, qty_seg)
                w     = wape_numpy(y_seg, final)
                rows_seg.append({"segment": seg, "threshold": thr, "wape": w})
                if w < best_w:
                    best_w, best_thr = w, thr
            best_thresholds[seg] = best_thr
            print(f"  [{seg:6s}]  threshold={best_thr:.3f}  WAPE={best_w:.4f}")
 
        pd.DataFrame(rows_seg).to_csv("/tmp/threshold_sweep_seg.csv", index=False)
        mlflow.log_artifact("/tmp/threshold_sweep_seg.csv")
        mlflow.log_params({f"threshold_{k}": v for k, v in best_thresholds.items()})
 
    except NameError:
        print("[INFO] model_qty non disponible — threshold sweep à faire après Stage 2.")
        best_thresholds = {"low": 0.5, "medium": 0.5, "high": 0.5}
 
    # COMMAND ----------
 
    # MAGIC %md
    # MAGIC ## 9. Diagnostic : PR-curve + Confusion Matrix + SHAP
 
    # COMMAND ----------
 
    # --- PR Curve ---
    precision, recall, pr_thr = precision_recall_curve(z_va, p_zero_val)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    axes[0].plot(recall, precision, lw=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title(f"PR Curve (AUC-PR={auc_pr:.3f})")
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3)
 
    # Confusion matrix au threshold 0.5 (calibré)
    pred_bin = (p_zero_val > 0.5).astype(int)
    cm       = confusion_matrix(z_va, pred_bin)
    im       = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Prédit non-zéro", "Prédit zéro"])
    axes[1].set_yticklabels(["Réel non-zéro", "Réel zéro"])
    axes[1].set_title("Confusion Matrix (thr=0.5)")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.tight_layout()
    fig.savefig("/tmp/zero_clf_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact("/tmp/zero_clf_diagnostics.png")
 
    # --- SHAP ---
    try:
        sample_idx = np.random.choice(len(X_va), size=min(3000, len(X_va)), replace=False)
        X_shap = X_va.iloc[sample_idx].copy()
        for c in FEATURES_CATEGORICAL:
            if c in X_shap.columns:
                X_shap[c] = X_shap[c].astype(float)
 
        explainer   = shap.TreeExplainer(model_zero)
        shap_values = explainer.shap_values(X_shap)
 
        # summary_plot retourne une liste [class0, class1] pour binary
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, show=False, max_display=25)
        fig.savefig("/tmp/shap_zero_clf.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        mlflow.log_artifact("/tmp/shap_zero_clf.png")
 
        # Top features SHAP (mean |SHAP|)
        shap_df = pd.DataFrame({
            "feature":    X_shap.columns,
            "mean_shap":  np.abs(sv).mean(axis=0),
        }).sort_values("mean_shap", ascending=False)
        shap_df.to_csv("/tmp/shap_zero_importance.csv", index=False)
        mlflow.log_artifact("/tmp/shap_zero_importance.csv")
        print("\nTop 10 features (SHAP) :")
        print(shap_df.head(10).to_string(index=False))
 
    except Exception as e:
        print(f"[WARN] SHAP skipped: {e}")

# COMMAND ----------

X_tr_clean = X_tr.head(5).copy()
    for c in FEATURES_CATEGORICAL:
        if c in X_tr_clean.columns:
            X_tr_clean[c] = X_tr_clean[c].astype(int)
 
    sig_zero = infer_signature(X_tr_clean, model_zero.predict(X_tr.head(5)))
    mlflow.lightgbm.log_model(
        model_zero,
        artifact_path="zero_classifier_v2",
        registered_model_name=MLFLOW_MODEL_NAME_ZERO,
        signature=sig_zero,
        input_example=X_tr_clean.head(1),
    )
 
    # Feature importance gain (en complément de SHAP)
    fi_zero = pd.DataFrame({
        "feature": FEATURES_ZERO,
        "gain":    model_zero.feature_importance(importance_type="gain"),
        "split":   model_zero.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi_zero.to_csv("/tmp/fi_zero_v2.csv", index=False)
    mlflow.log_artifact("/tmp/fi_zero_v2.csv")
 
    print(f"\n✅ Classifieur zéro v2 entraîné et loggé.")
    print(f"   AUC-ROC : {auc_roc:.4f}")
    print(f"   AUC-PR  : {auc_pr:.4f}")
    print(f"   Logloss : {logloss:.4f}")
    print(f"   Thresholds WAPE : {best_thresholds}")
