# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Train iterative two-stage LightGBM
# MAGIC
# MAGIC Training still uses fully known train rows. Validation early stopping is
# MAGIC now computed as a blind recursive forecast: validation targets are never
# MAGIC used as lag/rolling inputs, and each predicted week feeds the next one.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %pip install lightgbm==4.3.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import gc
import sys
sys.path.append("./")

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from src.iterative import (
    build_iterative_state,
    prepare_lgb_features,
    score_iterative_horizon,
)
from src.utils import wape_numpy

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load gold splits

# COMMAND ----------

META_COLS = ["semaine", "week_id", "annee", "num_sem", "code_agence", "code_article", "quantite", "is_dead_pair"]
cols_needed = list(dict.fromkeys(META_COLS + FEATURES))

train_pd = spark.table(TBL_GOLD_TRAIN).select(*cols_needed).toPandas()
val_pd = spark.table(TBL_GOLD_VAL).select(*cols_needed).toPandas()
internal_test_rows = spark.table(TBL_GOLD_INTERNAL_TEST).count()

for df in [train_pd, val_pd]:
    df["quantite"] = pd.to_numeric(df["quantite"], errors="coerce")
    df["week_id"] = df["week_id"].astype("int32")
    df["annee"] = df["annee"].astype("int32")
    df["num_sem"] = df["num_sem"].astype("int32")

print(f"Train: {len(train_pd):,}   Val: {len(val_pd):,}   Internal test: {internal_test_rows:,}")
assert train_pd["week_id"].max() < val_pd["week_id"].min(), "Temporal split broken"
assert val_pd["week_id"].max() < INTERNAL_TEST_START_WEEK_ID, "Temporal split broken"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare LightGBM matrices and iterative states

# COMMAND ----------

X_tr = prepare_lgb_features(train_pd, FEATURES, FEATURES_CATEGORICAL)
y_tr = train_pd["quantite"].astype("float64").to_numpy()
z_tr = (y_tr == 0).astype("int8")

nz_tr = y_tr > 0
X_tr_nz = X_tr.loc[nz_tr].reset_index(drop=True)
y_tr_nz = y_tr[nz_tr]

base_state_val = build_iterative_state(train_pd)
base_state_test = build_iterative_state(pd.concat([train_pd, val_pd], ignore_index=True))

print(f"Zero rate train: {z_tr.mean():.3f}")
print(f"Non-zero training rows: {len(X_tr_nz):,}")

dtrain_z = lgb.Dataset(
    X_tr,
    label=z_tr,
    categorical_feature=FEATURES_CATEGORICAL,
    free_raw_data=False,
)
dtrain_q = lgb.Dataset(
    X_tr_nz,
    label=y_tr_nz,
    categorical_feature=FEATURES_CATEGORICAL,
    free_raw_data=False,
)

qty_params = dict(LGB_PARAMS_QTY)
qty_params["objective"] = "regression_l1"
qty_params["metric"] = "None"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Iterative early stopping

# COMMAND ----------

def _copy_booster(booster):
    return lgb.Booster(model_str=booster.model_to_string())


def _iteration_checkpoints(max_rounds: int, checkpoints):
    out = sorted({int(c) for c in checkpoints if int(c) <= int(max_rounds)})
    if not out:
        out = [int(max_rounds)]
    return out


def _current_iter(booster) -> int:
    return int(booster.current_iteration())


def _iterative_wape(zero_booster, qty_booster, zero_iter, qty_iter, threshold, blend_alpha=0.0):
    pred_val = score_iterative_horizon(
        base_state=base_state_val,
        horizon_df=val_pd,
        zero_model=zero_booster,
        qty_model=qty_booster,
        features=FEATURES,
        categorical_features=FEATURES_CATEGORICAL,
        threshold=float(threshold),
        label="validation",
        zero_iter=zero_iter,
        qty_iter=qty_iter,
        blend_alpha=float(blend_alpha),
        feed_rounded=ITERATIVE_FEED_ROUNDED,
    )
    return wape_numpy(pred_val["quantite"].to_numpy(), pred_val["prediction"].to_numpy())


def _signature_frame(X):
    out = X.copy()
    for c in FEATURES_CATEGORICAL:
        if c in out.columns:
            if str(out[c].dtype) == "category":
                out[c] = out[c].cat.codes.astype("int32")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-1).astype("int32")
    for c in out.columns:
        if c not in FEATURES_CATEGORICAL:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


zero_checkpoints = _iteration_checkpoints(LGB_NUM_ROUNDS_ZERO, ZERO_ITER_CHECKPOINTS)
qty_checkpoints = _iteration_checkpoints(LGB_NUM_ROUNDS_QTY, QTY_ITER_CHECKPOINTS)

with mlflow.start_run(run_name="train_pipeline") as run:
    mlflow.log_params({
        "n_features": len(FEATURES),
        "train_rows": len(train_pd),
        "val_rows": len(val_pd),
        "internal_test_rows": int(internal_test_rows),
        "train_end": TRAIN_END_WEEK_ID,
        "val_end": VAL_END_WEEK_ID,
        "internal_test_start": INTERNAL_TEST_START_WEEK_ID,
        "internal_test_end": INTERNAL_TEST_END_WEEK_ID,
        "iterative_feed_rounded": ITERATIVE_FEED_ROUNDED,
        "zero_iter_checkpoints": str(zero_checkpoints),
        "qty_iter_checkpoints": str(qty_checkpoints),
        "iterative_early_stop_patience": ITERATIVE_EARLY_STOP_PATIENCE,
        "iterative_early_stop_min_delta": ITERATIVE_EARLY_STOP_MIN_DELTA,
        "warmup_qty_rounds_for_zero": WARMUP_QTY_ROUNDS_FOR_ZERO,
        "early_stop_threshold_grid": str(ITERATIVE_EARLY_STOP_THRESHOLD_GRID),
        "final_threshold_grid": str(ZERO_THRESHOLD_GRID),
        "blend_alpha_grid": str(PREDICTION_BLEND_ALPHA_GRID),
    })

    # The zero model needs a quantity model to turn p_zero into WAPE. Keep this
    # warm-up deliberately short: it is only a stable proxy for zero selection.
    warmup_qty_rounds = min(int(WARMUP_QTY_ROUNDS_FOR_ZERO), int(LGB_NUM_ROUNDS_QTY))
    print(f"Training fixed warm-up qty model for zero early stopping: {warmup_qty_rounds} rounds")
    warmup_qty_model = lgb.train(
        qty_params,
        dtrain_q,
        num_boost_round=warmup_qty_rounds,
        keep_training_booster=True,
    )

    model_zero = None
    current_zero_iter = 0
    best_zero_row = {"val_wape": float("inf")}
    best_zero_model = None
    no_improve_zero = 0

    for target_iter in zero_checkpoints:
        additional_rounds = int(target_iter) - current_zero_iter
        if additional_rounds <= 0:
            continue
        print(f"Training zero: rounds {current_zero_iter + 1}..{target_iter}")
        model_zero = lgb.train(
            LGB_PARAMS_ZERO,
            dtrain_z,
            num_boost_round=additional_rounds,
            init_model=model_zero,
            keep_training_booster=True,
        )
        current_zero_iter = _current_iter(model_zero)

        checkpoint_best = {"val_wape": float("inf")}
        for threshold in ITERATIVE_EARLY_STOP_THRESHOLD_GRID:
            val_wape = _iterative_wape(
                model_zero,
                warmup_qty_model,
                current_zero_iter,
                _current_iter(warmup_qty_model),
                threshold,
            )
            print(
                f"zero zi={current_zero_iter:4d} fixed_qi={_current_iter(warmup_qty_model):4d} "
                f"thr={threshold:.2f} -> iterative val WAPE={val_wape:.5f}"
            )
            if val_wape < checkpoint_best["val_wape"]:
                checkpoint_best = {"val_wape": val_wape, "threshold": float(threshold)}

        if checkpoint_best["val_wape"] + ITERATIVE_EARLY_STOP_MIN_DELTA < best_zero_row["val_wape"]:
            best_zero_row = {
                **checkpoint_best,
                "zero_iter": current_zero_iter,
                "qty_iter": _current_iter(warmup_qty_model),
            }
            best_zero_model = _copy_booster(model_zero)
            no_improve_zero = 0
            print(f"new best zero model at {current_zero_iter} rounds; iterative val WAPE={checkpoint_best['val_wape']:.5f}")
        else:
            no_improve_zero += 1
            print(f"zero no improvement {no_improve_zero}/{ITERATIVE_EARLY_STOP_PATIENCE}; best={best_zero_row['val_wape']:.5f}")

        if no_improve_zero >= ITERATIVE_EARLY_STOP_PATIENCE:
            print("Zero model iterative early stopping triggered.")
            break

    if best_zero_model is None:
        raise RuntimeError("No zero model selected during iterative early stopping.")

    del warmup_qty_model
    gc.collect()

    model_qty = None
    current_qty_iter = 0
    best_qty_row = {"val_wape": float("inf")}
    best_qty_model = None
    no_improve_qty = 0

    for target_iter in qty_checkpoints:
        additional_rounds = int(target_iter) - current_qty_iter
        if additional_rounds <= 0:
            continue
        print(f"Training qty: rounds {current_qty_iter + 1}..{target_iter}")
        model_qty = lgb.train(
            qty_params,
            dtrain_q,
            num_boost_round=additional_rounds,
            init_model=model_qty,
            keep_training_booster=True,
        )
        current_qty_iter = _current_iter(model_qty)

        checkpoint_best = {"val_wape": float("inf")}
        for threshold in ITERATIVE_EARLY_STOP_THRESHOLD_GRID:
            val_wape = _iterative_wape(
                best_zero_model,
                model_qty,
                int(best_zero_row["zero_iter"]),
                current_qty_iter,
                threshold,
            )
            print(
                f"qty fixed_zi={int(best_zero_row['zero_iter']):4d} qi={current_qty_iter:4d} "
                f"thr={threshold:.2f} -> iterative val WAPE={val_wape:.5f}"
            )
            if val_wape < checkpoint_best["val_wape"]:
                checkpoint_best = {"val_wape": val_wape, "threshold": float(threshold)}

        if checkpoint_best["val_wape"] + ITERATIVE_EARLY_STOP_MIN_DELTA < best_qty_row["val_wape"]:
            best_qty_row = {
                **checkpoint_best,
                "zero_iter": int(best_zero_row["zero_iter"]),
                "qty_iter": current_qty_iter,
            }
            best_qty_model = _copy_booster(model_qty)
            no_improve_qty = 0
            print(f"new best qty model at {current_qty_iter} rounds; iterative val WAPE={checkpoint_best['val_wape']:.5f}")
        else:
            no_improve_qty += 1
            print(f"qty no improvement {no_improve_qty}/{ITERATIVE_EARLY_STOP_PATIENCE}; best={best_qty_row['val_wape']:.5f}")

        if no_improve_qty >= ITERATIVE_EARLY_STOP_PATIENCE:
            print("Qty model iterative early stopping triggered.")
            break

    if best_qty_model is None:
        raise RuntimeError("No quantity model selected during iterative early stopping.")

    BEST_ZERO_ITER = int(best_qty_row["zero_iter"])
    BEST_QTY_ITER = int(best_qty_row["qty_iter"])

    # Final cheap tuning: test threshold 0.50 and 0.55, plus a small blend
    # against seasonal/recent history to dampen recursive drift.
    blend_rows = []
    for threshold in ZERO_THRESHOLD_GRID:
        for alpha in PREDICTION_BLEND_ALPHA_GRID:
            pred_val = score_iterative_horizon(
                base_state=base_state_val,
                horizon_df=val_pd,
                zero_model=best_zero_model,
                qty_model=best_qty_model,
                features=FEATURES,
                categorical_features=FEATURES_CATEGORICAL,
                threshold=float(threshold),
                label="validation",
                zero_iter=BEST_ZERO_ITER,
                qty_iter=BEST_QTY_ITER,
                blend_alpha=float(alpha),
                feed_rounded=ITERATIVE_FEED_ROUNDED,
            )
            val_wape = wape_numpy(pred_val["quantite"].to_numpy(), pred_val["prediction"].to_numpy())
            blend_rows.append({"threshold": float(threshold), "blend_alpha": float(alpha), "val_wape": val_wape})
            print(f"threshold={threshold:.2f} blend alpha={alpha:.2f} -> iterative val WAPE={val_wape:.5f}")
            del pred_val
            gc.collect()

    threshold_df = pd.DataFrame(blend_rows).sort_values("val_wape")
    best_selection = threshold_df.iloc[0]
    BEST_THRESHOLD = float(best_selection["threshold"])
    BEST_BLEND_ALPHA = float(best_selection["blend_alpha"])
    BEST_VAL_WAPE = float(best_selection["val_wape"])

    print(
        f"Selected zero_iter={BEST_ZERO_ITER}, qty_iter={BEST_QTY_ITER}, "
        f"threshold={BEST_THRESHOLD:.2f}, blend_alpha={BEST_BLEND_ALPHA:.2f}, "
        f"iterative validation WAPE={BEST_VAL_WAPE:.5f}"
    )

    mlflow.log_param("best_iteration_zero", BEST_ZERO_ITER)
    mlflow.log_param("best_iteration_qty", BEST_QTY_ITER)
    mlflow.log_param("best_zero_threshold", BEST_THRESHOLD)
    mlflow.log_param("best_blend_alpha", BEST_BLEND_ALPHA)
    mlflow.log_metric("val_wape", BEST_VAL_WAPE)

    threshold_df.to_csv("/tmp/threshold_blend_sweep.csv", index=False)
    mlflow.log_artifact("/tmp/threshold_blend_sweep.csv")

    # Log models without input_example. MLflow validates input examples with
    # plain integer categoricals, which can trigger LightGBM's category mismatch.
    sig_n = min(100, len(X_tr))
    sig_zero = infer_signature(
        _signature_frame(X_tr.head(sig_n)),
        best_zero_model.predict(X_tr.head(sig_n), num_iteration=BEST_ZERO_ITER),
    )
    mlflow.lightgbm.log_model(
        best_zero_model,
        artifact_path="zero_classifier",
        registered_model_name=MLFLOW_MODEL_NAME_ZERO,
        signature=sig_zero,
    )

    sig_q_n = min(100, len(X_tr_nz))
    sig_qty = infer_signature(
        _signature_frame(X_tr_nz.head(sig_q_n)),
        best_qty_model.predict(X_tr_nz.head(sig_q_n), num_iteration=BEST_QTY_ITER),
    )
    mlflow.lightgbm.log_model(
        best_qty_model,
        artifact_path="qty_regressor",
        registered_model_name=MLFLOW_MODEL_NAME_QTY,
        signature=sig_qty,
    )

    fi_zero = pd.DataFrame({
        "feature": FEATURES,
        "gain": best_zero_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    fi_qty = pd.DataFrame({
        "feature": FEATURES,
        "gain": best_qty_model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    fi_zero.to_csv("/tmp/fi_zero.csv", index=False)
    fi_qty.to_csv("/tmp/fi_qty.csv", index=False)
    mlflow.log_artifact("/tmp/fi_zero.csv")
    mlflow.log_artifact("/tmp/fi_qty.csv")

    # -------- Iterative validation and internal test predictions --------
    val_iter = score_iterative_horizon(
        base_state=base_state_val,
        horizon_df=val_pd,
        zero_model=best_zero_model,
        qty_model=best_qty_model,
        features=FEATURES,
        categorical_features=FEATURES_CATEGORICAL,
        threshold=BEST_THRESHOLD,
        label="validation",
        zero_iter=BEST_ZERO_ITER,
        qty_iter=BEST_QTY_ITER,
        blend_alpha=BEST_BLEND_ALPHA,
        feed_rounded=ITERATIVE_FEED_ROUNDED,
    )
    val_wape = wape_numpy(val_iter["quantite"].to_numpy(), val_iter["prediction"].to_numpy())

    test_pd = spark.table(TBL_GOLD_INTERNAL_TEST).select(*cols_needed).toPandas()
    test_pd["quantite"] = pd.to_numeric(test_pd["quantite"], errors="coerce")
    test_pd["week_id"] = test_pd["week_id"].astype("int32")
    test_pd["annee"] = test_pd["annee"].astype("int32")
    test_pd["num_sem"] = test_pd["num_sem"].astype("int32")

    test_iter = score_iterative_horizon(
        base_state=base_state_test,
        horizon_df=test_pd,
        zero_model=best_zero_model,
        qty_model=best_qty_model,
        features=FEATURES,
        categorical_features=FEATURES_CATEGORICAL,
        threshold=BEST_THRESHOLD,
        label="internal_test",
        zero_iter=BEST_ZERO_ITER,
        qty_iter=BEST_QTY_ITER,
        blend_alpha=BEST_BLEND_ALPHA,
        feed_rounded=ITERATIVE_FEED_ROUNDED,
    )
    test_wape = wape_numpy(test_iter["quantite"].to_numpy(), test_iter["prediction"].to_numpy())

    mlflow.log_metric("val_wape_iterative_final", val_wape)
    mlflow.log_metric("internal_test_wape", test_wape)
    print(f"Validation iterative WAPE: {val_wape:.5f}")
    print(f"Internal test iterative WAPE: {test_wape:.5f}")

    schema = StructType([
        StructField("semaine", StringType()),
        StructField("code_agence", LongType()),
        StructField("code_article", LongType()),
        StructField("quantite", LongType()),
        StructField("p_zero", DoubleType()),
        StructField("qty_pred", DoubleType()),
        StructField("prediction", DoubleType()),
    ])

    def _write_prediction_table(pred_df: pd.DataFrame, table_name: str):
        out = pred_df[["semaine", "code_agence", "code_article", "quantite", "p_zero", "qty_pred", "prediction"]].copy()
        out["code_agence"] = out["code_agence"].astype("int64")
        out["code_article"] = out["code_article"].astype("int64")
        out["quantite"] = out["quantite"].fillna(0).round().astype("int64")
        sdf = spark.createDataFrame(out, schema=schema)
        (
            sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name)
        )
        print(f"Wrote {len(out):,} rows to {table_name}")

    _write_prediction_table(val_iter, TBL_VAL_PREDICTIONS)
    _write_prediction_table(test_iter, TBL_INTERNAL_TEST_PREDICTIONS)
