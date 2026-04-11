# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Configuration
# MAGIC
# MAGIC Single source of truth for the SGDB 2026 forecasting pipeline.
# MAGIC
# MAGIC **Purpose:** every magic number, table name, split date, feature list, and
# MAGIC hyperparameter lives here. Every downstream notebook imports this file via
# MAGIC `%run ./00_config` so that a single change propagates through the whole
# MAGIC pipeline.
# MAGIC
# MAGIC **Inputs:** none.
# MAGIC **Outputs:** Python variables available in the parent notebook.
# MAGIC
# MAGIC **Assumption:** the notebook is attached to a Databricks cluster where
# MAGIC `workspace.default` is writable and the hackathon tables are already
# MAGIC provisioned.

# COMMAND ----------

# -----------------------------------------------------------------------------
# Team identity & submission table
# -----------------------------------------------------------------------------
NOM_EQUIPE = "télécacaton"
TABLE_PREDICTIONS = f"workspace.default.predictions_equipe_{NOM_EQUIPE}"

# -----------------------------------------------------------------------------
# Source tables (provisioned by the hackathon)
# -----------------------------------------------------------------------------
TBL_TRAIN = "workspace.default.histo_ventes_train"
TBL_TEST = "workspace.default.histo_ventes_test"
TBL_AGENCE = "workspace.default.donnees_agence"
TBL_ARTICLES = "workspace.default.donnees_articles"
TBL_FACTURATION = "workspace.default.donnees_facturation"

# -----------------------------------------------------------------------------
# Delta tables produced by the Lakeflow (DLT) pipeline
# -----------------------------------------------------------------------------
# These live in `workspace.default` and are materialised by
# `pipeline_lakeflow.py`. Downstream notebooks read from them with
# `spark.table(...)`.
TBL_SILVER_VENTES = "workspace.default.silver_ventes"
TBL_SILVER_PANEL = "workspace.default.silver_panel"
TBL_GOLD_FEATURES = "workspace.default.gold_feature_table"
TBL_GOLD_TRAIN = "workspace.default.gold_train"
TBL_GOLD_VAL = "workspace.default.gold_validation"
TBL_GOLD_INTERNAL_TEST = "workspace.default.gold_internal_test"
TBL_GOLD_FINAL_INFERENCE = "workspace.default.gold_test_features"
# Backward-compatible alias. This table is the hidden final-inference horizon,
# not the labelled internal test set.
TBL_GOLD_TEST = TBL_GOLD_FINAL_INFERENCE

# ML-notebook-owned tables (not produced by DLT):
TBL_PREDICTIONS_FINAL = "workspace.default.predictions_final"
TBL_VAL_PREDICTIONS = "workspace.default.val_predictions"
TBL_INTERNAL_TEST_PREDICTIONS = "workspace.default.internal_test_predictions"

# -----------------------------------------------------------------------------
# Temporal splits
# -----------------------------------------------------------------------------
# Train on everything up to 2024-W26.
# Validate on 2024-W27 .. 2024-W52 with the validation horizon masked when
# building lag/rolling features.
# Internal labelled test: 2025-W01 .. 2025-W26, also masked at feature time.
# Final inference / leaderboard submission: 2025-W27 .. 2025-W52.
TRAIN_END_WEEK_ID = 202426
VAL_START_WEEK_ID = 202427
VAL_END_WEEK_ID = 202452
INTERNAL_TEST_START_WEEK_ID = 202501
INTERNAL_TEST_END_WEEK_ID = 202526
FINAL_INFERENCE_START_WEEK_ID = 202527
FINAL_INFERENCE_END_WEEK_ID = 202552

# Short aliases kept for older notebook text; use the explicit names in new code.
TEST_START_WEEK_ID = INTERNAL_TEST_START_WEEK_ID
TEST_END_WEEK_ID = INTERNAL_TEST_END_WEEK_ID

# -----------------------------------------------------------------------------
# MLflow experiment
# -----------------------------------------------------------------------------
MLFLOW_EXPERIMENT = f"/Shared/sgdb2026_{NOM_EQUIPE}"
MLFLOW_MODEL_NAME_ZERO = f"workspace.default.sgdb2026_{NOM_EQUIPE}_zero_classifier"
MLFLOW_MODEL_NAME_QTY = f"workspace.default.sgdb2026_{NOM_EQUIPE}_quantity_regressor"

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Data cleaning knobs
# -----------------------------------------------------------------------------
# Per-pair outlier capping. We clip each pair's series at its own P99.5 to
# knock down the top 0.5% of spikes (the tail goes to 21 646 with a P99 of
# 179 — see section 2 of the EDA report). This is robust to heterogeneous
# pairs because the quantile is computed per (agence, article).
OUTLIER_PERCENTILE = 0.995

# A week is "anomalous" if its quantity is > ANOMALY_MULTIPLIER × the pair's
# rolling-26 median. Those rows are flagged, not dropped — the model sees the
# flag as a feature and the (capped) value as the target.
ANOMALY_MULTIPLIER = 10.0
ANOMALY_ROLL_WINDOW = 26

# Light EMA smoothing used to build an auxiliary denoised target. The model
# trains on the raw target; the smoothed one is only exposed as a feature.
SMOOTH_ALPHA = 0.3

# -----------------------------------------------------------------------------
# Feature engineering knobs
# -----------------------------------------------------------------------------
# Lags we compute. Only lags ≥ 26 will be available at inference time because
# we predict W27..W52 in a single shot — lags 1..25 would need future data
# from within the prediction window. We still keep them for training because
# they help the model learn short-range patterns, and the inference notebook
# fills them with expanding statistics instead.
LAGS_ALL = [1, 2, 4, 8, 13, 26, 52, 104]
LAGS_INFERENCE_SAFE = [26, 52, 104]

# Rolling window sizes (weeks), each used for both mean and std.
ROLLING_WINDOWS = [4, 8, 13, 26, 52]

# Median is more robust on heavy-tailed series; we compute it on 4 and 13
# weeks only to keep the shuffle cost bounded.
ROLLING_MEDIAN_WINDOWS = [4, 13]

# -----------------------------------------------------------------------------
# Feature list fed to the models
# -----------------------------------------------------------------------------
# Keep this list flat. Every string here must be present in the feature table
# built by 02_feature_engineering.py. The training notebook asserts this.
FEATURES_NUMERIC = [
    # --- Lags ---
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52", "lag_104",
    # --- Rolling mean ---
    "roll_mean_4", "roll_mean_8", "roll_mean_13", "roll_mean_26", "roll_mean_52",
    # --- Rolling std ---
    "roll_std_4", "roll_std_8", "roll_std_13", "roll_std_26", "roll_std_52",
    # --- Rolling median (robust) ---
    "roll_median_4", "roll_median_13",
    # --- Zero rates & trend ---
    "zero_rate_26", "zero_rate_52", "pair_zero_rate_expanding",
    "trend_8", "ratio_n1_vs_mean", "yoy_ratio",
    # --- Pair expanding stats ---
    "pair_mean", "pair_median", "pair_max", "pair_count", "pair_cv",
    # --- Same-week-of-year history ---
    "sem_mean", "sem_max", "sem_median",
    # --- Agency expanding stats ---
    "agence_mean", "agence_median",
    # --- Article expanding stats ---
    "article_mean", "article_median",
    # --- Activity ---
    "n_active_weeks",
    # --- Billing-derived (monthly granularity, lagged 2 months) ---
    "fac_prix_unit", "fac_pct_pro", "fac_nb_chantiers", "fac_nb_achats",
    # --- Temporal ---
    "annee", "num_sem", "sin_sem", "cos_sem",
    # --- Calendar flags ---
    "is_summer_trough", "is_xmas_trough",
]

FEATURES_CATEGORICAL = [
    "art_specialite_enc",
    "art_famille_enc",
    "art_marque_enc",
    "art_mdd_enc",
    "ag_region_enc",
]

FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

# -----------------------------------------------------------------------------
# LightGBM hyperparameters — Stage 1: zero classifier
# -----------------------------------------------------------------------------
# 67% of rows have quantite = 0, so class weight is close to the default but
# we still give a slight boost to the positive class (non-zero) because those
# are the rows that drive WAPE (WAPE = sum|err| / sum|y|, and y is 0 for the
# negative class).
LGB_PARAMS_ZERO = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.5,
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}
LGB_NUM_ROUNDS_ZERO = 3000
LGB_EARLY_STOP_ZERO = 75
# Minimum improvement (in logloss) required to count as progress. Without
# this, the classifier keeps training through 0.0001-sized improvements
# that are indistinguishable from val noise.
LGB_MIN_DELTA_ZERO = 1e-3

# -----------------------------------------------------------------------------
# LightGBM hyperparameters — Stage 2: quantity regressor
# -----------------------------------------------------------------------------
# Tweedie is a strong default for zero-heavy non-negative targets; we still
# train on log1p(quantite) because the evaluation pipeline tries both and
# picks the lower-WAPE variant. Sample weights are proportional to the
# quantity itself so the regressor focuses on high-volume pairs — the same
# pairs that dominate WAPE.
LGB_PARAMS_QTY = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "metric": "None",   # we use custom WAPE
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "max_bin": 511,
    "n_jobs": -1,
    "seed": SEED,
    "verbose": -1,
}
LGB_NUM_ROUNDS_QTY = 5000
LGB_EARLY_STOP_QTY = 100
# Minimum improvement (in WAPE units) to count as progress. 1e-3 stops the
# regressor as soon as the 50-iter gain falls below a milliquème — the
# training curve plateaus to that scale around iter 800 on this dataset.
LGB_MIN_DELTA_QTY = 1e-3

# -----------------------------------------------------------------------------
# Zero threshold search (legacy single-threshold grid)
# -----------------------------------------------------------------------------
# Kept for reference; the new training pipeline uses the adaptive grid below.
ZERO_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# -----------------------------------------------------------------------------
# Ensemble / stacking additions (new model pipeline)
# -----------------------------------------------------------------------------
# These blocks are consumed exclusively by `03_train_model.py` and
# `05_inference.py`. They have no effect on the feature engineering layer.

# Multi-seed ensembling. Each listed seed trains one extra LGB clf / LGB reg /
# XGB reg, and predictions are averaged before stacking. Default is a single
# seed (fast); bump to `[42, 1337, 2024]` for the final run — cost scales
# linearly with len(ENSEMBLE_SEEDS).
ENSEMBLE_SEEDS = [SEED]

# XGBoost regressor — added alongside LGB as a diversity term in the ensemble.
# Same Tweedie objective so scales are comparable with LGB.
XGB_PARAMS_QTY = {
    "objective":              "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "eval_metric":            "mae",
    "learning_rate":          0.03,
    "max_depth":              8,
    "min_child_weight":       10.0,
    "subsample":              0.8,
    "colsample_bytree":       0.8,
    "reg_alpha":              0.1,
    "reg_lambda":             1.0,
    "gamma":                  0.0,
    "verbosity":              0,
    "nthread":                -1,
    "seed":                   SEED,
}
XGB_NUM_ROUNDS_QTY = 5000
XGB_EARLY_STOP_QTY = 100
XGB_MIN_DELTA_QTY  = 1e-3

# NOTE — sample weights were tried for WAPE alignment (§5.6 of the report)
# but produced catastrophic predictions on Tweedie: variance∝μ^p already
# weights high-volume rows, so sample_weight=y double-counts and makes the
# model systematically over-predict (val WAPE >3 instead of <1). Dropped.
# If you want WAPE-aligned weighting, switch the objective to regression_l1
# first — MAE is homoscedastic and accepts y-proportional weights cleanly.

# Ridge stacker — blends (lgb_pred, xgb_pred, lag_52, pair_mean, pair_median).
# `positive=True` and `fit_intercept=False` preserve zero-homogeneity and
# keep every meta-weight non-negative.
STACK_BASELINE_COLS = ["lag_52", "pair_mean", "pair_median"]
STACKING_RIDGE_ALPHA = 0.5

# Stacker fit strategy.
#
# - 0 (default) → fit Ridge on validation predictions from the main models.
#                 This matches the Fourth-good-model notebook and is the
#                 ROBUST choice when train spans multiple years with regime
#                 shifts (2020-2024 here). Minimal overfit risk: 5 features
#                 on ~85k non-zero rows.
# - >= 2        → TimeSeriesSplit OOF on the training set (was the default
#                 briefly). AVOID unless train is stationary: on this dataset,
#                 OOF folds train on earlier, calmer periods and the fold
#                 models under-predict vs the full-train model → Ridge learns
#                 a 1.22x correction that then over-amplifies the main-model
#                 predictions on val, blowing WAPE from ~0.80 to ~0.91.
STACKING_OOF_FOLDS = 0

# Isotonic calibration of the zero classifier. Fitted on validation probas
# vs actual `is_zero`, applied at inference to stabilise the adaptive threshold.
CALIBRATE_ZERO_CLF = True

# Adaptive zero threshold: thr(row) = clip(base - slope * pair_zero_rate, clip_lo, clip_hi).
# Products with a high historical zero-rate get a lower threshold (more
# aggressive zeroing). The (base, slope) pair is picked by a small grid
# sweep on val WAPE — NOT a full hyperparameter search.
ADAPTIVE_THRESHOLD_BASE_GRID  = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
ADAPTIVE_THRESHOLD_SLOPE_GRID = [0.00, 0.10, 0.15, 0.20]
ADAPTIVE_THRESHOLD_CLIP       = (0.15, 0.85)
ZERO_RATE_FEATURE             = "pair_zero_rate_expanding"

# Hard override: pairs flagged `is_dead_pair == 1` are forced to zero after
# blending. Belt-and-braces on top of the classifier; cheap and safe.
DEAD_PAIR_FORCE_ZERO = True

# MLflow artifact folder name. The training notebook writes all models +
# stacker + calibrator + a `manifest.json` under this sub-path; inference
# downloads the same folder and rebuilds the pipeline in-process. This
# replaces the Model Registry flow for the ensemble case because the
# Registry does not elegantly represent k-model ensembles.
MLFLOW_ENSEMBLE_ARTIFACT_PATH = "ensemble"

# -----------------------------------------------------------------------------
# Echo
# -----------------------------------------------------------------------------
print(f"Team            : {NOM_EQUIPE}")
print(f"Submission tbl  : {TABLE_PREDICTIONS}")
print(f"Train           : week_id <= {TRAIN_END_WEEK_ID}")
print(f"Validation      : [{VAL_START_WEEK_ID}, {VAL_END_WEEK_ID}]")
print(f"Internal test   : [{INTERNAL_TEST_START_WEEK_ID}, {INTERNAL_TEST_END_WEEK_ID}]")
print(f"Final inference : [{FINAL_INFERENCE_START_WEEK_ID}, {FINAL_INFERENCE_END_WEEK_ID}]")
print(f"#features       : {len(FEATURES)}  ({len(FEATURES_NUMERIC)} num, {len(FEATURES_CATEGORICAL)} cat)")
print(f"Seed            : {SEED}")
