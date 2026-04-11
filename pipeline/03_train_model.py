# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Train, select features, register the two-stage model
# MAGIC
# MAGIC This notebook trains a wide two-stage LightGBM on all candidate features,
# MAGIC scores feature relevance with gain, permutation importance and block
# MAGIC ablations, then iteratively shrinks the feature set to an elite subset of
# MAGIC roughly 75 batch-safe features. The final registered models are trained on
# MAGIC the selected subset.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install lightgbm==4.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import gc
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.append("./")

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from src.utils import wape_lgb_feval, wape_numpy

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

ID_COLS = ["semaine", "code_agence", "code_article", "week_id", "quantite", "is_dead_pair"]
TMP_DIR = "/tmp"


def assert_columns_exist(sdf, required_cols: Sequence[str], label: str) -> None:
    missing = [c for c in required_cols if c not in sdf.columns]
    if missing:
        preview = ", ".join(missing[:20])
        raise RuntimeError(f"{label} is missing {len(missing)} required columns: {preview}")


def build_selection_sample(sdf, max_rows: int):
    counts = (
        sdf.agg(
            F.count("*").alias("total_rows"),
            F.sum((F.col("quantite") > 0).cast("long")).alias("positive_rows"),
        )
        .collect()[0]
    )
    total_rows = int(counts["total_rows"])
    positive_rows = int(counts["positive_rows"] or 0)
    zero_rows = total_rows - positive_rows

    if total_rows <= max_rows:
        return sdf, {
            "selection_total_rows": total_rows,
            "selection_positive_rows": positive_rows,
            "selection_zero_rows": zero_rows,
            "selection_sampled": False,
        }

    positive_target = min(positive_rows, int(max_rows * 0.70))
    zero_target = max_rows - positive_target
    pos_frac = min(1.0, positive_target / max(positive_rows, 1))
    zero_frac = min(1.0, zero_target / max(zero_rows, 1))

    sampled = (
        sdf.filter(F.col("quantite") > 0).sample(False, pos_frac, SEED)
        .unionByName(
            sdf.filter(F.col("quantite") == 0).sample(False, zero_frac, SEED + 1)
        )
    )
    return sampled, {
        "selection_total_rows": total_rows,
        "selection_positive_rows": positive_rows,
        "selection_zero_rows": zero_rows,
        "selection_sampled": True,
        "selection_pos_frac": float(pos_frac),
        "selection_zero_frac": float(zero_frac),
        "selection_max_rows": int(max_rows),
    }


def collect_to_pandas(sdf, features: Sequence[str], label: str) -> pd.DataFrame:
    cols = ID_COLS + list(features)
    assert_columns_exist(sdf, cols, label)
    pdf = sdf.select(*cols).toPandas()

    pdf["quantite"] = pd.to_numeric(pdf["quantite"], errors="coerce").astype(np.float32)
    pdf["is_dead_pair"] = pd.to_numeric(pdf["is_dead_pair"], errors="coerce").fillna(0).astype(np.int8)
    pdf["week_id"] = pd.to_numeric(pdf["week_id"], errors="coerce").astype(np.int32)

    cat_set = set(FEATURES_CATEGORICAL)
    for c in features:
        if c in cat_set:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(-1).astype(np.int32)
        else:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce").astype(np.float32)
    print(f"{label}: {len(pdf):,} rows x {len(features):,} features")
    return pdf


def build_xy(df: pd.DataFrame, features: Sequence[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = df.loc[:, list(features)].copy()
    y = df["quantite"].astype(np.float32).to_numpy()
    is_zero = (y == 0).astype(np.int8)
    return X, y, is_zero


def categorical_for(features: Sequence[str]) -> List[str]:
    feature_set = set(features)
    return [c for c in FEATURES_CATEGORICAL if c in feature_set]


def train_zero_model(
    X_train,
    z_train,
    X_val,
    z_val,
    features: Sequence[str],
    num_rounds: int,
    early_stop: int,
    log_period: int,
):
    cats = categorical_for(features)
    dtrain = lgb.Dataset(X_train, label=z_train, categorical_feature=cats)
    dval = lgb.Dataset(X_val, label=z_val, reference=dtrain, categorical_feature=cats)
    return lgb.train(
        LGB_PARAMS_ZERO,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stop, first_metric_only=True),
            lgb.log_evaluation(period=log_period),
        ],
    )


def train_qty_model(
    X_train,
    y_train,
    X_val,
    y_val,
    features: Sequence[str],
    num_rounds: int,
    early_stop: int,
    log_period: int,
):
    nz_train = y_train > 0
    nz_val = y_val > 0
    if int(nz_train.sum()) == 0 or int(nz_val.sum()) == 0:
        raise RuntimeError("Quantity regressor needs non-zero rows in train and validation.")

    cats = categorical_for(features)
    dtrain = lgb.Dataset(
        X_train.loc[nz_train],
        label=y_train[nz_train],
        categorical_feature=cats,
    )
    dval = lgb.Dataset(
        X_val.loc[nz_val],
        label=y_val[nz_val],
        reference=dtrain,
        categorical_feature=cats,
    )

    qty_params = dict(LGB_PARAMS_QTY)
    qty_params["objective"] = "regression_l1"
    qty_params["metric"] = "None"

    return lgb.train(
        qty_params,
        dtrain,
        num_boost_round=num_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        feval=wape_lgb_feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stop, first_metric_only=True),
            lgb.log_evaluation(period=log_period),
        ],
    )


def score_two_stage(model_zero, model_qty, X_val, y_val, threshold=None) -> Dict[str, object]:
    p_zero = model_zero.predict(X_val, num_iteration=model_zero.best_iteration)
    qty_pred = np.clip(model_qty.predict(X_val, num_iteration=model_qty.best_iteration), 0.0, None)

    if threshold is None:
        best_threshold = None
        best_wape = float("inf")
        rows = []
        for thr in ZERO_THRESHOLD_GRID:
            final = np.where(p_zero > thr, 0.0, qty_pred)
            wape = wape_numpy(y_val, final)
            rows.append({"threshold": float(thr), "wape": float(wape)})
            if wape < best_wape:
                best_wape = float(wape)
                best_threshold = float(thr)
        threshold_df = pd.DataFrame(rows)
    else:
        best_threshold = float(threshold)
        final = np.where(p_zero > best_threshold, 0.0, qty_pred)
        best_wape = float(wape_numpy(y_val, final))
        threshold_df = pd.DataFrame([{"threshold": best_threshold, "wape": best_wape}])

    final = np.where(p_zero > best_threshold, 0.0, qty_pred)
    return {
        "p_zero": p_zero,
        "qty_pred": qty_pred,
        "prediction": final,
        "best_threshold": best_threshold,
        "wape": best_wape,
        "threshold_df": threshold_df,
    }


def fit_two_stage(
    X_train,
    y_train,
    z_train,
    X_val,
    y_val,
    z_val,
    features: Sequence[str],
    num_rounds_zero: int,
    num_rounds_qty: int,
    early_stop_zero: int,
    early_stop_qty: int,
    log_period: int = 200,
):
    model_zero = train_zero_model(
        X_train,
        z_train,
        X_val,
        z_val,
        features,
        num_rounds_zero,
        early_stop_zero,
        log_period,
    )
    model_qty = train_qty_model(
        X_train,
        y_train,
        X_val,
        y_val,
        features,
        num_rounds_qty,
        early_stop_qty,
        log_period,
    )
    scored = score_two_stage(model_zero, model_qty, X_val, y_val)
    return model_zero, model_qty, scored


def combined_gain_importance(model_zero, model_qty, features: Sequence[str]) -> pd.DataFrame:
    fi = pd.DataFrame({
        "feature": list(features),
        "gain_zero": model_zero.feature_importance(importance_type="gain"),
        "gain_qty": model_qty.feature_importance(importance_type="gain"),
    })
    max_zero = max(float(fi["gain_zero"].max()), 1e-12)
    max_qty = max(float(fi["gain_qty"].max()), 1e-12)
    fi["gain_zero_norm"] = fi["gain_zero"] / max_zero
    fi["gain_qty_norm"] = fi["gain_qty"] / max_qty
    fi["combined_gain"] = 0.45 * fi["gain_zero_norm"] + 0.55 * fi["gain_qty_norm"]
    return fi.sort_values("combined_gain", ascending=False).reset_index(drop=True)


def permutation_importance(
    model_zero,
    model_qty,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold: float,
    features_to_test: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = score_two_stage(model_zero, model_qty, X_val, y_val, threshold=threshold)["wape"]
    X_work = X_val.copy()
    rows = []
    for i, feature in enumerate(features_to_test, start=1):
        original = X_work[feature].to_numpy(copy=True)
        X_work[feature] = rng.permutation(original)
        wape = score_two_stage(model_zero, model_qty, X_work, y_val, threshold=threshold)["wape"]
        X_work[feature] = original
        rows.append({
            "feature": feature,
            "perm_wape": float(wape),
            "perm_delta_wape": float(wape - base),
        })
        if i % 25 == 0:
            print(f"Permutation importance: {i}/{len(features_to_test)} features")
    return pd.DataFrame(rows).sort_values("perm_delta_wape", ascending=False)


def block_permutation_importance(
    model_zero,
    model_qty,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    threshold: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = score_two_stage(model_zero, model_qty, X_val, y_val, threshold=threshold)["wape"]
    X_work = X_val.copy()
    rows = []
    for block, block_features in FEATURE_BLOCKS.items():
        cols = [c for c in block_features if c in X_work.columns]
        if not cols:
            continue
        original = X_work.loc[:, cols].copy()
        perm_idx = rng.permutation(len(X_work))
        for c in cols:
            X_work[c] = original[c].to_numpy()[perm_idx]
        wape = score_two_stage(model_zero, model_qty, X_work, y_val, threshold=threshold)["wape"]
        X_work.loc[:, cols] = original
        rows.append({
            "block": block,
            "n_features": len(cols),
            "block_perm_wape": float(wape),
            "block_delta_wape": float(wape - base),
        })
        print(f"Block permutation {block}: delta WAPE {wape - base:.5f}")
    return pd.DataFrame(rows).sort_values("block_delta_wape", ascending=False)


def normalise_positive(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=0).fillna(0.0)
    max_val = float(clipped.max())
    if max_val <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return clipped / max_val


def feature_block_lookup() -> Dict[str, str]:
    lookup = {}
    for block, block_features in FEATURE_BLOCKS.items():
        for f in block_features:
            lookup[f] = block
    return lookup


def build_feature_score_table(
    gain_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    block_df: pd.DataFrame,
) -> pd.DataFrame:
    block_lookup = feature_block_lookup()
    block_delta = dict(zip(block_df["block"], block_df["block_delta_wape"]))
    safe_set = set(FEATURES_BATCH_SAFE)

    out = gain_df.merge(
        perm_df[["feature", "perm_delta_wape"]],
        on="feature",
        how="left",
    )
    out["block"] = out["feature"].map(block_lookup).fillna("unknown")
    out["block_delta_wape"] = out["block"].map(block_delta).fillna(0.0)
    out["is_batch_safe"] = out["feature"].isin(safe_set)
    out["perm_delta_wape"] = out["perm_delta_wape"].fillna(0.0)
    out["gain_score"] = normalise_positive(out["combined_gain"])
    out["perm_score"] = normalise_positive(out["perm_delta_wape"])
    out["block_score"] = normalise_positive(out["block_delta_wape"])
    out["elite_score"] = (
        0.55 * out["gain_score"]
        + 0.35 * out["perm_score"]
        + 0.10 * out["block_score"]
    )
    return out.sort_values(
        ["is_batch_safe", "elite_score", "combined_gain"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def save_and_log_csv(df: pd.DataFrame, filename: str) -> str:
    path = os.path.join(TMP_DIR, filename)
    df.to_csv(path, index=False)
    mlflow.log_artifact(path)
    return path


def evaluate_elimination_path(
    ordered_features: Sequence[str],
    X_train_all: pd.DataFrame,
    y_train: np.ndarray,
    z_train: np.ndarray,
    X_val_all: pd.DataFrame,
    y_val: np.ndarray,
    z_val: np.ndarray,
) -> pd.DataFrame:
    steps = sorted(
        {
            min(step, len(ordered_features))
            for step in FEATURE_SELECTION_STEPS
            if step >= TARGET_FEATURE_COUNT
        },
        reverse=True,
    )
    if TARGET_FEATURE_COUNT not in steps:
        steps.append(TARGET_FEATURE_COUNT)

    rows = []
    for k in steps:
        features_k = list(ordered_features[:k])
        print(f"Elimination step: training with top {k} features")
        model_zero, model_qty, scored = fit_two_stage(
            X_train_all.loc[:, features_k],
            y_train,
            z_train,
            X_val_all.loc[:, features_k],
            y_val,
            z_val,
            features_k,
            FEATURE_SELECTION_NUM_ROUNDS_ZERO,
            FEATURE_SELECTION_NUM_ROUNDS_QTY,
            FEATURE_SELECTION_EARLY_STOP_ZERO,
            FEATURE_SELECTION_EARLY_STOP_QTY,
            log_period=250,
        )
        rows.append({
            "n_features": k,
            "wape": scored["wape"],
            "threshold": scored["best_threshold"],
            "best_iteration_zero": model_zero.best_iteration,
            "best_iteration_qty": model_qty.best_iteration,
        })
        del model_zero, model_qty, scored
        gc.collect()
    return pd.DataFrame(rows).sort_values("n_features", ascending=False)


# COMMAND ----------

train_raw_sdf = spark.table(TBL_GOLD_TRAIN)
val_raw_sdf = spark.table(TBL_GOLD_VAL)
required_cols = ID_COLS + FEATURES
assert_columns_exist(train_raw_sdf, required_cols, "gold_train")
assert_columns_exist(val_raw_sdf, required_cols, "gold_validation")

train_sdf = train_raw_sdf.select(*required_cols)
val_sdf = val_raw_sdf.select(*required_cols)
selection_train_sdf, sample_stats = build_selection_sample(train_sdf, FEATURE_SELECTION_MAX_TRAIN_ROWS)

print(f"Candidate features: {len(FEATURES)}")
print(f"Batch-safe candidates: {len(FEATURES_BATCH_SAFE)}")
print(sample_stats)

# COMMAND ----------

train_sel_pd = collect_to_pandas(selection_train_sdf, FEATURES, "selection_train")
val_pd = collect_to_pandas(val_sdf, FEATURES, "validation")

X_sel_all, y_sel, z_sel = build_xy(train_sel_pd, FEATURES)
X_val_all, y_val, z_val = build_xy(val_pd, FEATURES)

print(f"Zero rate selection train: {z_sel.mean():.3f}   val: {z_val.mean():.3f}")
assert train_sel_pd["week_id"].max() < val_pd["week_id"].min(), "Temporal split broken"

# COMMAND ----------

with mlflow.start_run(run_name="train_pipeline") as parent_run:
    mlflow.log_params({
        "n_candidate_features": len(FEATURES),
        "n_batch_safe_features": len(FEATURES_BATCH_SAFE),
        "target_feature_count": TARGET_FEATURE_COUNT,
        "train_rows_total": int(sample_stats["selection_total_rows"]),
        "selection_train_rows": len(train_sel_pd),
        "val_rows": len(val_pd),
        "train_end": VAL_START_WEEK_ID,
        "val_end": VAL_END_WEEK_ID,
        "selection_sampled": bool(sample_stats["selection_sampled"]),
    })
    for k, v in sample_stats.items():
        if k not in {"selection_sampled"} and isinstance(v, (int, float, bool)):
            mlflow.log_param(k, v)

    # ------------------------------------------------------------------
    # 1. Wide screening model on all 250 features.
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name="feature_screening_all_250", nested=True):
        model_zero_all, model_qty_all, scored_all = fit_two_stage(
            X_sel_all,
            y_sel,
            z_sel,
            X_val_all,
            y_val,
            z_val,
            FEATURES,
            FEATURE_SELECTION_NUM_ROUNDS_ZERO,
            FEATURE_SELECTION_NUM_ROUNDS_QTY,
            FEATURE_SELECTION_EARLY_STOP_ZERO,
            FEATURE_SELECTION_EARLY_STOP_QTY,
            log_period=250,
        )
        mlflow.log_metric("raw_val_wape_all_features", scored_all["wape"])
        mlflow.log_param("raw_best_threshold_all_features", scored_all["best_threshold"])
        mlflow.log_param("best_iteration_zero_all_features", model_zero_all.best_iteration)
        mlflow.log_param("best_iteration_qty_all_features", model_qty_all.best_iteration)

        gain_df = combined_gain_importance(model_zero_all, model_qty_all, FEATURES)
        save_and_log_csv(gain_df, "feature_gain_all_250.csv")
        save_and_log_csv(scored_all["threshold_df"], "threshold_sweep_all_250.csv")

        if len(val_pd) > PERMUTATION_MAX_VAL_ROWS:
            perm_idx = np.random.default_rng(SEED).choice(
                len(val_pd),
                size=PERMUTATION_MAX_VAL_ROWS,
                replace=False,
            )
            X_perm = X_val_all.iloc[perm_idx].reset_index(drop=True)
            y_perm = y_val[perm_idx]
        else:
            X_perm = X_val_all.reset_index(drop=True)
            y_perm = y_val

        top_gain_candidates = gain_df["feature"].head(PERMUTATION_MAX_FEATURES // 2).tolist()
        top_safe_candidates = (
            gain_df[gain_df["feature"].isin(FEATURES_BATCH_SAFE)]["feature"]
            .head(PERMUTATION_MAX_FEATURES)
            .tolist()
        )
        perm_candidates = list(dict.fromkeys(top_gain_candidates + top_safe_candidates))[:PERMUTATION_MAX_FEATURES]
        perm_df = permutation_importance(
            model_zero_all,
            model_qty_all,
            X_perm,
            y_perm,
            scored_all["best_threshold"],
            perm_candidates,
            seed=SEED + 10,
        )
        block_df = block_permutation_importance(
            model_zero_all,
            model_qty_all,
            X_perm,
            y_perm,
            scored_all["best_threshold"],
            seed=SEED + 20,
        )
        save_and_log_csv(perm_df, "feature_permutation_importance.csv")
        save_and_log_csv(block_df, "feature_block_ablation.csv")

    score_df = build_feature_score_table(gain_df, perm_df, block_df)
    save_and_log_csv(score_df, "feature_selection_scores.csv")
    del model_zero_all, model_qty_all, scored_all, X_perm, y_perm
    gc.collect()

    safe_ordered_features = score_df.loc[score_df["is_batch_safe"], "feature"].tolist()
    if len(safe_ordered_features) < TARGET_FEATURE_COUNT:
        safe_ordered_features = score_df["feature"].tolist()
        mlflow.log_param("batch_safe_fallback_used", True)
    else:
        mlflow.log_param("batch_safe_fallback_used", False)

    # ------------------------------------------------------------------
    # 2. Iterative elimination path on the screening sample.
    # ------------------------------------------------------------------
    with mlflow.start_run(run_name="iterative_elimination", nested=True):
        elimination_df = evaluate_elimination_path(
            safe_ordered_features,
            X_sel_all,
            y_sel,
            z_sel,
            X_val_all,
            y_val,
            z_val,
        )
        save_and_log_csv(elimination_df, "feature_elimination_path.csv")
        for _, row in elimination_df.iterrows():
            mlflow.log_metric(f"wape_top_{int(row['n_features'])}", float(row["wape"]))

    selected_features = list(safe_ordered_features[:TARGET_FEATURE_COUNT])
    selected_cat_features = categorical_for(selected_features)
    selected_df = pd.DataFrame({
        "rank": np.arange(1, len(selected_features) + 1),
        "feature": selected_features,
    }).merge(score_df, on="feature", how="left")
    save_and_log_csv(selected_df, "selected_features_final_75.csv")

    mlflow.log_param("selected_feature_count", len(selected_features))
    mlflow.log_param("selected_categorical_count", len(selected_cat_features))
    mlflow.log_metric(
        "selected_features_mean_elite_score",
        float(selected_df["elite_score"].fillna(0.0).mean()),
    )

    # Free the wide training sample before collecting the full train set.
    del train_sel_pd, X_sel_all, y_sel, z_sel
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Final full training on the selected 75 features.
    # ------------------------------------------------------------------
    train_final_pd = collect_to_pandas(train_sdf, selected_features, "final_train")
    X_train_final, y_train_final, z_train_final = build_xy(train_final_pd, selected_features)
    X_val_final = X_val_all.loc[:, selected_features].copy()
    val_meta = val_pd[["semaine", "code_agence", "code_article", "quantite"]].copy()
    del X_val_all, val_pd
    gc.collect()

    with mlflow.start_run(run_name="stage1_zero_clf_final", nested=True):
        model_zero = train_zero_model(
            X_train_final,
            z_train_final,
            X_val_final,
            z_val,
            selected_features,
            LGB_NUM_ROUNDS_ZERO,
            LGB_EARLY_STOP_ZERO,
            log_period=100,
        )
        p_zero_val = model_zero.predict(X_val_final, num_iteration=model_zero.best_iteration)
        val_logloss = float(
            -np.mean(
                z_val * np.log(np.clip(p_zero_val, 1e-7, 1 - 1e-7))
                + (1 - z_val) * np.log(np.clip(1 - p_zero_val, 1e-7, 1 - 1e-7))
            )
        )
        mlflow.log_metric("val_logloss", val_logloss)
        mlflow.log_param("best_iteration_zero", model_zero.best_iteration)
        sig_zero = infer_signature(
            X_train_final.head(5),
            model_zero.predict(X_train_final.head(5), num_iteration=model_zero.best_iteration),
        )
        mlflow.lightgbm.log_model(
            model_zero,
            artifact_path="zero_classifier",
            registered_model_name=MLFLOW_MODEL_NAME_ZERO,
            signature=sig_zero,
            input_example=X_train_final.head(1),
        )
        final_fi_zero = pd.DataFrame({
            "feature": selected_features,
            "gain": model_zero.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        save_and_log_csv(final_fi_zero, "fi_zero_final.csv")

    with mlflow.start_run(run_name="stage2_qty_regressor_final", nested=True):
        model_qty = train_qty_model(
            X_train_final,
            y_train_final,
            X_val_final,
            y_val,
            selected_features,
            LGB_NUM_ROUNDS_QTY,
            LGB_EARLY_STOP_QTY,
            log_period=100,
        )
        mlflow.log_param("best_iteration_qty", model_qty.best_iteration)
        sig_qty = infer_signature(
            X_train_final.loc[y_train_final > 0].head(5),
            model_qty.predict(
                X_train_final.loc[y_train_final > 0].head(5),
                num_iteration=model_qty.best_iteration,
            ),
        )
        mlflow.lightgbm.log_model(
            model_qty,
            artifact_path="qty_regressor",
            registered_model_name=MLFLOW_MODEL_NAME_QTY,
            signature=sig_qty,
            input_example=X_train_final.loc[y_train_final > 0].head(1),
        )
        final_fi_qty = pd.DataFrame({
            "feature": selected_features,
            "gain": model_qty.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        save_and_log_csv(final_fi_qty, "fi_qty_final.csv")

    final_scored = score_two_stage(model_zero, model_qty, X_val_final, y_val)
    threshold_df = final_scored["threshold_df"]
    print(threshold_df)
    print(f"Best threshold: {final_scored['best_threshold']} -> WAPE = {final_scored['wape']:.4f}")

    mlflow.log_param("best_zero_threshold", final_scored["best_threshold"])
    mlflow.log_param("best_iteration_zero", model_zero.best_iteration)
    mlflow.log_param("best_iteration_qty", model_qty.best_iteration)
    mlflow.log_metric("val_wape", final_scored["wape"])
    save_and_log_csv(threshold_df, "threshold_sweep_final_75.csv")

    # Validation predictions for 04_evaluate.py.
    val_out = val_meta.copy()
    val_out["code_agence"] = pd.to_numeric(val_out["code_agence"], errors="coerce").astype(np.int64)
    val_out["code_article"] = pd.to_numeric(val_out["code_article"], errors="coerce").astype(np.int64)
    val_out["quantite"] = pd.to_numeric(val_out["quantite"], errors="coerce").round().astype(np.int64)
    val_out["p_zero"] = final_scored["p_zero"]
    val_out["qty_pred"] = final_scored["qty_pred"]
    val_out["prediction"] = final_scored["prediction"]

    schema = StructType([
        StructField("semaine", StringType()),
        StructField("code_agence", LongType()),
        StructField("code_article", LongType()),
        StructField("quantite", LongType()),
        StructField("p_zero", DoubleType()),
        StructField("qty_pred", DoubleType()),
        StructField("prediction", DoubleType()),
    ])
    val_pred_sdf = spark.createDataFrame(val_out, schema=schema)
    (
        val_pred_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TBL_VAL_PREDICTIONS)
    )
    print(f"Wrote validation predictions to {TBL_VAL_PREDICTIONS}")

    del train_final_pd, X_train_final, X_val_final, val_meta
    gc.collect()
