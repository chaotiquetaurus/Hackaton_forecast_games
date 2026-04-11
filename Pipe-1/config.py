# -----------------------------------------------------------------------------
# Configuration — Single source of truth for the SGDB 2026 pipeline
# -----------------------------------------------------------------------------

# Source tables (provisioned by the hackathon)
TBL_TRAIN = "workspace.default.histo_ventes_train"
TBL_TEST = "workspace.default.histo_ventes_test"
TBL_AGENCE = "workspace.default.donnees_agence"
TBL_ARTICLES = "workspace.default.donnees_articles"
TBL_FACTURATION = "workspace.default.donnees_facturation"

# Pair key reused by every windowed feature
PAIR_KEYS = ["code_agence", "code_article"]

# Temporal splits
# Known-label train: up to 2024-W26.
# Validation: 2024-W27 .. 2024-W52, with its horizon masked for lag features.
# Internal labelled test: 2025-W01 .. 2025-W26, masked the same way.
# Final inference / leaderboard: 2025-W27 .. 2025-W52.
TRAIN_END_WEEK_ID = 202426
VAL_START_WEEK_ID = 202427
VAL_END_WEEK_ID = 202452
INTERNAL_TEST_START_WEEK_ID = 202501
INTERNAL_TEST_END_WEEK_ID = 202526
FINAL_INFERENCE_START_WEEK_ID = 202527
FINAL_INFERENCE_END_WEEK_ID = 202552

# Data cleaning knobs
OUTLIER_PERCENTILE = 0.995
ANOMALY_MULTIPLIER = 10.0
ANOMALY_ROLL_WINDOW = 26

# Feature engineering knobs
LAGS_ALL = [1, 2, 4, 8, 13, 26, 52, 104]
ROLLING_WINDOWS = [4, 8, 13, 26, 52]
ROLLING_MEDIAN_WINDOWS = [4, 13]

# Feature list
FEATURES_NUMERIC = [
    "lag_1", "lag_2", "lag_4", "lag_8", "lag_13", "lag_26", "lag_52", "lag_104",
    "roll_mean_4", "roll_mean_8", "roll_mean_13", "roll_mean_26", "roll_mean_52",
    "roll_std_4", "roll_std_8", "roll_std_13", "roll_std_26", "roll_std_52",
    "roll_median_4", "roll_median_13",
    "zero_rate_26", "zero_rate_52", "pair_zero_rate_expanding",
    "trend_8", "ratio_n1_vs_mean", "yoy_ratio",
    "pair_mean", "pair_median", "pair_max", "pair_count", "pair_cv",
    "sem_mean", "sem_max", "sem_median",
    "agence_mean", "agence_median",
    "article_mean", "article_median",
    "n_active_weeks",
    "fac_prix_unit", "fac_pct_pro", "fac_nb_chantiers", "fac_nb_achats",
    "annee", "num_sem", "sin_sem", "cos_sem",
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
