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
import shap
import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
 
train_pd = train_pd[train_pd["is_dead_pair"] == 0].reset_index(drop=True)
print(f"Train: {len(train_pd):,}   Val: {len(val_pd):,}")
 

# COMMAND ----------

def add_zero_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Taux de zéro par article
    art_zero = (
        ref_df.groupby("code_article")["quantite"]
        .apply(lambda x: (x == 0).mean())
        .rename("zero_rate_article").reset_index()
    )
    df = df.merge(art_zero, on="code_article", how="left")
    df["zero_rate_article"] = df["zero_rate_article"].fillna(0.5)
 
    # Taux de zéro par agence
    ag_zero = (
        ref_df.groupby("code_agence")["quantite"]
        .apply(lambda x: (x == 0).mean())
        .rename("zero_rate_agence").reset_index()
    )
    df = df.merge(ag_zero, on="code_agence", how="left")
    df["zero_rate_agence"] = df["zero_rate_agence"].fillna(0.5)
 
    # Stats par paire
    pair_stats = (
        ref_df.groupby(["code_agence", "code_article"])["quantite"]
        .agg(
            zero_rate_pair = lambda x: (x == 0).mean(),
            mean_qty_pair  = "mean",
            std_qty_pair   = "std",
        )
        .reset_index()
    )
    pair_stats["cv_pair"] = (
        pair_stats["std_qty_pair"] / (pair_stats["mean_qty_pair"] + 1e-6)
    ).clip(0, 10)
    pair_stats["is_always_zero_pair"] = (pair_stats["zero_rate_pair"] == 1.0).astype(int)
 
    df = df.merge(
        pair_stats[["code_agence", "code_article", "zero_rate_pair", "cv_pair", "is_always_zero_pair"]],
        on=["code_agence", "code_article"], how="left",
    )
    df["zero_rate_pair"]     = df["zero_rate_pair"].fillna(0.5)
    df["cv_pair"]            = df["cv_pair"].fillna(1.0)
    df["is_always_zero_pair"]= df["is_always_zero_pair"].fillna(0)
 
    # Taux de zéro récent (8 dernières semaines)
    max_week  = ref_df["week_id"].max()
    recent    = ref_df[ref_df["week_id"] >= max_week - 7]
    rec_zero  = (
        recent.groupby(["code_agence", "code_article"])["quantite"]
        .apply(lambda x: (x == 0).mean())
        .rename("zero_rate_pair_recent").reset_index()
    )
    df = df.merge(rec_zero, on=["code_agence", "code_article"], how="left")
    df["zero_rate_pair_recent"] = df["zero_rate_pair_recent"].fillna(df["zero_rate_pair"])
 
    # Semaines depuis la dernière vente
    last_sale = (
        ref_df[ref_df["quantite"] > 0]
        .groupby(["code_agence", "code_article"])["week_id"]
        .max().rename("last_sale_week_id").reset_index()
    )
    df = df.merge(last_sale, on=["code_agence", "code_article"], how="left")
    df["weeks_since_last_sale"] = (
        df["week_id"] - df["last_sale_week_id"].fillna(df["week_id"] - 52)
    ).clip(0, 104)
    df = df.drop(columns=["last_sale_week_id"])
 
    # Interaction : paire intermittente ET récemment à zéro
    df["intermittent_and_recent_zero"] = df["cv_pair"] * df["zero_rate_pair_recent"]
 
    return df
 
ZERO_FEATURES_ADDED = [
    "zero_rate_article",
    "zero_rate_agence",
    "zero_rate_pair",
    "cv_pair",
    "is_always_zero_pair",
    "zero_rate_pair_recent",
    "weeks_since_last_sale",
    "intermittent_and_recent_zero",
]
 
train_pd = add_zero_features(train_pd, ref_df=train_pd)
val_pd   = add_zero_features(val_pd,   ref_df=train_pd)
 
FEATURES_ZERO = FEATURES + ZERO_FEATURES_ADDED
print(f"Features: {len(FEATURES_ZERO)}  (+{len(ZERO_FEATURES_ADDED)} zero-signal)")
 

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare X / y / is_zero

# COMMAND ----------

def build_xy_zero(df, features):
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
 
# Poids par article (WAPE-aligned)
article_vol = train_pd.groupby("code_article")["quantite"].sum().rename("art_vol")
train_pd = train_pd.join(article_vol, on="code_article")
val_pd   = val_pd.join(article_vol,   on="code_article")
val_pd["art_vol"] = val_pd["art_vol"].fillna(1.0)
 
w_tr = 1.0 / (train_pd["art_vol"].values + 1.0)
w_va = 1.0 / (val_pd["art_vol"].values   + 1.0)
 

# COMMAND ----------

n_pos = int(z_tr.sum())
n_neg = int((z_tr == 0).sum())
print(f"n_zero={n_pos:,}  n_nonzero={n_neg:,}  ratio={n_neg/n_pos:.2f}")
 
# Paramètres communs aux deux phases
BASE_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    # Capacité volontairement limitée
    "num_leaves":        31,
    "max_depth":         6,
    "min_child_samples": 200,      # clé anti-overfit
    # Régularisation
    "reg_alpha":         1.0,      # L1 fort
    "reg_lambda":        5.0,      # L2 fort
    "min_split_gain":    0.1,
    "min_child_weight":  1e-3,
    # Bagging agressif
    "subsample":         0.6,
    "subsample_freq":    1,
    "colsample_bytree":  0.6,
    # Pas de scale_pos_weight : géré par les poids sample
    "is_unbalance":      False,
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
 
with mlflow.start_run(run_name="stage1_zero_clf_v3") as run:
    mlflow.log_params({
        **BASE_PARAMS,
        "n_features_zero":    len(FEATURES_ZERO),
        "train_rows":         len(train_pd),
        "val_rows":           len(val_pd),
        "zero_rate_train":    float(z_tr.mean()),
        "zero_rate_val":      float(z_va.mean()),
    })
 
    # free_raw_data=False obligatoire pour init_model en phase 2
    dtrain_z = lgb.Dataset(
        X_tr, label=z_tr, weight=w_tr,
        categorical_feature=FEATURES_CATEGORICAL,
        free_raw_data=False,
    )
    dval_z = lgb.Dataset(
        X_va, label=z_va, weight=w_va,
        reference=dtrain_z,
        categorical_feature=FEATURES_CATEGORICAL,
        free_raw_data=False,
    )
 
    # ---- Phase 1 : exploration (LR modéré, early stopping court) ----
    print("=== Phase 1 : exploration ===")
    params_p1 = {**BASE_PARAMS, "learning_rate": 0.05}
 
    model_zero = lgb.train(
        params_p1,
        dtrain_z,
        num_boost_round=500,
        valid_sets=[dtrain_z, dval_z],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Phase 1 terminée à l'itération {model_zero.best_iteration}")
 
    # ---- Phase 2 : affinage fin (LR très bas, early stopping patient) ----
    print("\n=== Phase 2 : affinage ===")
    params_p2 = {**BASE_PARAMS, "learning_rate": 0.005}
 
    model_zero = lgb.train(
        params_p2,
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
    print(f"Phase 2 terminée à l'itération {model_zero.best_iteration}")
    mlflow.log_param("best_iteration_zero", model_zero.best_iteration)
 

# COMMAND ----------

 p_zero_raw = model_zero.predict(X_va, num_iteration=model_zero.best_iteration)
 
    mid = len(p_zero_raw) // 2
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_zero_raw[:mid], z_va[:mid])
    p_zero_val = iso.transform(p_zero_raw)
 
    # Vérification : la calibration ne doit pas écraser la variance
    print(f"p_zero_raw  — mean={p_zero_raw.mean():.3f}  std={p_zero_raw.std():.3f}")
    print(f"p_zero_cal  — mean={p_zero_val.mean():.3f}  std={p_zero_val.std():.3f}")
 

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
    print(f"\nAUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  Logloss={logloss:.4f}")
 
    # Vérification overfit : le gap train/val ne doit pas dépasser ~0.05
    p_zero_train = model_zero.predict(X_tr, num_iteration=model_zero.best_iteration)
    logloss_train = float(
        -np.mean(
            z_tr * np.log(np.clip(p_zero_train, 1e-7, 1 - 1e-7))
            + (1 - z_tr) * np.log(np.clip(1 - p_zero_train, 1e-7, 1 - 1e-7))
        )
    )
    gap = logloss - logloss_train
    mlflow.log_metric("overfit_gap_logloss", gap)
    print(f"Logloss train={logloss_train:.4f}  val={logloss:.4f}  gap={gap:.4f}")
    if gap > 0.1:
        print("[WARN] Gap train/val > 0.1 — considérer augmenter min_child_samples ou reg_lambda")

# COMMAND ----------

try:
        _ = model_qty
        qty_pred = np.clip(
            model_qty.predict(X_va, num_iteration=model_qty.best_iteration), 0.0, None
        )
 
        vol_q33 = val_pd["art_vol"].quantile(0.33)
        vol_q66 = val_pd["art_vol"].quantile(0.66)
        val_pd["vol_segment"] = pd.cut(
            val_pd["art_vol"],
            bins=[-np.inf, vol_q33, vol_q66, np.inf],
            labels=["low", "medium", "high"],
        )
 
        best_thresholds, rows_seg = {}, []
        for seg in ["low", "medium", "high"]:
            mask    = val_pd["vol_segment"].values == seg
            p_seg   = p_zero_val[mask]
            qty_seg = qty_pred[mask]
            y_seg   = y_va[mask]
 
            best_thr, best_w = 0.5, float("inf")
            for thr in np.arange(0.05, 0.95, 0.025):
                final = np.where(p_seg > thr, 0.0, qty_seg)
                w     = wape_numpy(y_seg, final)
                rows_seg.append({"segment": seg, "threshold": thr, "wape": w})
                if w < best_w:
                    best_w, best_thr = w, thr
            best_thresholds[seg] = best_thr
            print(f"  [{seg:6s}]  threshold={best_thr:.3f}  WAPE={best_w:.4f}")
 
        pd.DataFrame(rows_seg).to_csv("/tmp/threshold_sweep_v3.csv", index=False)
        mlflow.log_artifact("/tmp/threshold_sweep_v3.csv")
        mlflow.log_params({f"threshold_{k}": v for k, v in best_thresholds.items()})
 
    except NameError:
        print("[INFO] model_qty non disponible — threshold sweep à faire après Stage 2.")
        best_thresholds = {"low": 0.5, "medium": 0.5, "high": 0.5}
 

# COMMAND ----------

 results = model_zero.evals_result()
    if "train" in results and "val" in results:
        iters       = range(len(results["train"]["binary_logloss"]))
        train_loss  = results["train"]["binary_logloss"]
        val_loss    = results["val"]["binary_logloss"]
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(iters, train_loss, label="Train")
        axes[0].plot(iters, val_loss,   label="Val")
        axes[0].axvline(model_zero.best_iteration, color="red", linestyle="--", label="Best iter")
        axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Logloss")
        axes[0].set_title("Train vs Val Logloss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # --- PR Curve ---
    precision, recall, _ = precision_recall_curve(z_va, p_zero_val)
    axes[1].plot(recall, precision, lw=2)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR Curve (AUC-PR={auc_pr:.3f})")
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1]); axes[1].grid(alpha=0.3)
 
    plt.tight_layout()
    fig.savefig("/tmp/zero_clf_v3_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact("/tmp/zero_clf_v3_curves.png")
 
    # --- SHAP ---
    try:
        idx    = np.random.choice(len(X_va), size=min(3000, len(X_va)), replace=False)
        X_shap = X_va.iloc[idx].copy()
        for c in FEATURES_CATEGORICAL:
            if c in X_shap.columns:
                X_shap[c] = X_shap[c].astype(float)
 
        explainer   = shap.TreeExplainer(model_zero)
        shap_values = explainer.shap_values(X_shap)
        sv          = shap_values[1] if isinstance(shap_values, list) else shap_values
 
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_shap, show=False, max_display=25)
        fig.savefig("/tmp/shap_zero_v3.png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        mlflow.log_artifact("/tmp/shap_zero_v3.png")
 
        shap_df = pd.DataFrame({
            "feature":   X_shap.columns,
            "mean_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_shap", ascending=False)
        shap_df.to_csv("/tmp/shap_zero_v3_importance.csv", index=False)
        mlflow.log_artifact("/tmp/shap_zero_v3_importance.csv")
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
        artifact_path="zero_classifier_v3",
        registered_model_name=MLFLOW_MODEL_NAME_ZERO,
        signature=sig_zero,
        input_example=X_tr_clean.head(1),
    )
 
    fi_zero = pd.DataFrame({
        "feature": FEATURES_ZERO,
        "gain":    model_zero.feature_importance(importance_type="gain"),
        "split":   model_zero.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi_zero.to_csv("/tmp/fi_zero_v3.csv", index=False)
    mlflow.log_artifact("/tmp/fi_zero_v3.csv")
 
    print(f"\n✅ Zero classifier v3")
    print(f"   AUC-ROC : {auc_roc:.4f}")
    print(f"   AUC-PR  : {auc_pr:.4f}")
    print(f"   Logloss : {logloss:.4f}")
    print(f"   Gap overfit : {gap:.4f}")
 
