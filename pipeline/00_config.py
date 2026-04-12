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
TBL_GOLD_TEST = "workspace.default.gold_test_features"

# ML-notebook-owned tables (not produced by DLT):
TBL_PREDICTIONS_FINAL = "workspace.default.predictions_final"
TBL_VAL_PREDICTIONS = "workspace.default.val_predictions"

# -----------------------------------------------------------------------------
# Temporal splits
# -----------------------------------------------------------------------------
# Train on everything strictly before 2025-W01.
# Validate on 2025-W01 .. 2025-W26 (first semester 2025).
# Predict (test) on 2025-W27 .. 2025-W52.
TRAIN_END_WEEK_ID = 202500   # train rows have week_id <  202501
VAL_START_WEEK_ID = 202501
VAL_END_WEEK_ID = 202526
TEST_START_WEEK_ID = 202527
TEST_END_WEEK_ID = 202552

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
    # --- Lags (inference-safe: >= 26) ---
    "lag_27", "lag_52", "lag_104",
    # --- Pair stats lag26 (inference-safe expanding stats) ---
    "pair_mean_lag26", "pair_median_lag26", "pair_max_lag26", "pair_min_lag26",
    "pair_std_lag26", "pair_cv_lag26",
    "pair_zero_rate_lag26", "pair_active_weeks_lag26",
    "pair_nonzero_mean_lag26", "pair_nonzero_share_lag26",
    "pair_last_nonzero_gap",
    # --- Ratios & trends ---
    "ratio_lag52_vs_pair_mean_lag26",
    "ratio_band26_52_vs_pair_mean_lag26",
    "trend_band_26_52_vs_52_104",
    "yoy_ratio",
    # --- Band features (inference-safe windowed aggregates) ---
    "band_mean_27_52",
    "band_std_26_52",
    "band_nonzero_mean_26_52", "band_nonzero_mean_52_104",
    "band_cv_26_52", "band_cv_27_52", "band_cv_39_65",
    "band_cv_52_104", "band_cv_78_104", "band_cv_104_156",
    "band_zero_rate_27_52", "band_zero_rate_39_65",
    "band_zero_rate_52_78", "band_zero_rate_52_104",
    "band_active_count_52_104", "band_active_count_104_156",
    # --- Same-week-of-year history ---
    "sem_nonzero_mean", "sem_zero_rate",
    # --- Agency stats lag26 ---
    "agence_mean_lag26", "agence_zero_rate_lag26", "agence_nonzero_mean_lag26",
    # --- Article stats lag26 ---
    "article_mean_lag26", "article_zero_rate_lag26", "article_nonzero_mean_lag26",
    # --- Billing-derived (rolling 6-month, lagged) ---
    "fac_nb_achats", "fac_achats_roll6",
    "fac_qty_roll6", "fac_montant_roll6", "fac_pct_pro_roll6",
    # --- Derived (computed by prepare_features in src/utils.py) ---
    "detrended_lag52",
    # --- Temporal ---
    "num_sem", "trimestre", "mois_approx", "week_of_quarter",
    "sin_sem", "cos_sem",
    "sin_sem_2", "cos_sem_2",
    "sin_sem_3", "cos_sem_3",
    "sin_sem_4", "cos_sem_4",
    # --- Calendar flags ---
    "is_summer_trough", "is_xmas_trough",
    "is_august", "is_december",
    "is_quarter_end", "is_peak_spring", "is_peak_autumn",
    "weeks_to_summer", "weeks_to_xmas",
]

FEATURES_CATEGORICAL = [
    "art_specialite_enc",
    "art_famille_enc",
    "art_sous_famille_enc",
    "art_marque_enc",
    "art_mdd_enc",
    "art_unite_vente_enc",
    "art_gamme_enc",
    "art_fournisseur_enc",
    "ag_region_enc",
    "ag_departement_enc",
    "ag_ville_enc",
    # Derived (computed by prepare_features in src/utils.py)
    "demand_profile",
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

# -----------------------------------------------------------------------------
# Optuna hyperparameter tuning
# -----------------------------------------------------------------------------
# Turn this off for a quick smoke run. For the final overnight job, keep it on:
# Optuna first searches better LightGBM parameters on the temporal validation
# split, then 03_train_model.py retrains the two final stages on the full train.
RUN_HYPERPARAMETER_TUNING = True

# Trial counts are intentionally asymmetric: the quantity regressor has the
# largest impact on WAPE once the zero gate is reasonably calibrated.
OPTUNA_N_TRIALS_ZERO = 50
OPTUNA_N_TRIALS_QTY = 80

# Optional hard time budgets. Leave as None to use the trial counts above.
OPTUNA_TIMEOUT_SECONDS_ZERO = None
OPTUNA_TIMEOUT_SECONDS_QTY = None

# Tuning rounds are shorter than final training rounds to keep search efficient.
OPTUNA_NUM_ROUNDS_ZERO = 1500
OPTUNA_NUM_ROUNDS_QTY = 2500
OPTUNA_EARLY_STOP_ZERO = 50
OPTUNA_EARLY_STOP_QTY = 80

# The zero classifier sees every row and is the most expensive to tune. A fixed
# random training sample keeps the search tractable while validation remains
# full-size and temporal. Set to None to tune on all rows.
OPTUNA_TUNING_MAX_TRAIN_ROWS_ZERO = 1_000_000
OPTUNA_TUNING_MAX_TRAIN_ROWS_QTY = None

# -----------------------------------------------------------------------------
# Zero threshold search
# -----------------------------------------------------------------------------
# After both stages run, we sweep the probability threshold applied to the
# zero classifier. Anything above it is forced to zero. The threshold that
# minimises WAPE on validation is persisted as an MLflow parameter and read
# back at inference time.
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

# XGBoost regressor — diversity term in the ensemble. Objective matches LGB
# (absolute error / MAE) so both models produce predictions on the same
# scale and the Ridge stacker can blend them without bias correction.
# reg:absoluteerror is available in XGBoost >= 2.1 (currently pinned 2.1.3).
XGB_PARAMS_QTY = {
    "objective":              "reg:absoluteerror",
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
#
# USE_STACKER toggle: when False, `blend = lgb_val_avg` directly (XGB and the
# baselines are skipped at blend time, though both models still train so the
# artifacts keep the same shape).
#
# DEFAULT = False because Ridge minimises MSE while the MAE-trained
# regressors predict the conditional MEDIAN. On right-skewed retail
# targets (mean/median ratio ~1.5) the Ridge systematically rescales LGB
# by that ratio, which forces the adaptive threshold into max-aggressive
# gating (kills ~78% of real sales) to recover. Net result on this data:
# final WAPE 0.92 with stacker vs ~0.80-0.85 expected without. Bypassing
# is correct here; flip to True only if you fix the objective mismatch
# (e.g. switch the Ridge to an L1-minimising blend, or retrain models to
# predict the mean).
USE_STACKER = False
STACK_BASELINE_COLS = ["lag_52", "pair_mean_lag26", "pair_median_lag26"]
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
# Base grid widened up to 0.80 because the pre-bypass runs systematically
# hit the lowest corner (0.40) — that was a symptom of stacker over-scaling,
# not a genuinely better threshold. With USE_STACKER=False the expectation
# is the optimum lands around 0.55-0.70.
ADAPTIVE_THRESHOLD_BASE_GRID  = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
ADAPTIVE_THRESHOLD_SLOPE_GRID = [0.00, 0.05, 0.10, 0.15, 0.20]
ADAPTIVE_THRESHOLD_CLIP       = (0.15, 0.90)
ZERO_RATE_FEATURE             = "pair_zero_rate_lag26"

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
print(f"Train end       : week_id < {VAL_START_WEEK_ID}")
print(f"Validation      : [{VAL_START_WEEK_ID}, {VAL_END_WEEK_ID}]")
print(f"Test (predict)  : [{TEST_START_WEEK_ID}, {TEST_END_WEEK_ID}]")
print(f"#features       : {len(FEATURES)}  ({len(FEATURES_NUMERIC)} num, {len(FEATURES_CATEGORICAL)} cat)")
print(f"Seed            : {SEED}")
