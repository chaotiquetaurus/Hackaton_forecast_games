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

# Data cleaning knobs
OUTLIER_PERCENTILE = 0.995
ANOMALY_MULTIPLIER = 10.0
ANOMALY_ROLL_WINDOW = 26

# Feature engineering knobs
#
# We deliberately generate a wide candidate set. The model pipeline will score
# every feature, then keep a compact elite subset for the final overnight model.
LAGS_ALL = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 13, 16, 20, 24,
    26, 27, 28, 30, 32, 35, 39, 44, 48, 51, 52, 53,
    56, 60, 65, 78, 91, 104, 130, 156, 208,
]
LAGS_INFERENCE_SAFE = [n for n in LAGS_ALL if n >= 26]
ROLLING_WINDOWS = [4, 8, 13, 26, 52]
ROLLING_MEDIAN_WINDOWS = [4, 13, 26, 52]
ROLLING_BASE_METRICS = [
    "mean", "std", "min", "max", "sum",
    "zero_rate", "active_count", "nonzero_mean", "cv",
]

# Lag-band windows are batch-safe for a 26-week forecast horizon because they
# only inspect observations at least 26 weeks before the row being scored.
LAG_BANDS = [
    (26, 52),
    (27, 52),
    (39, 65),
    (52, 78),
    (52, 104),
    (78, 104),
    (104, 156),
]
BAND_METRICS = [
    "mean", "std", "min", "max",
    "zero_rate", "active_count", "nonzero_mean", "cv",
]
PAIR_SAFE_GAP = 26

# Feature list: 250 candidates total.
FEATURES_LAGS = [f"lag_{n}" for n in LAGS_ALL]

FEATURES_ROLLING = [
    f"roll_{metric}_{window}"
    for window in ROLLING_WINDOWS
    for metric in ROLLING_BASE_METRICS
] + [f"roll_median_{window}" for window in ROLLING_MEDIAN_WINDOWS]

FEATURES_LAG_BANDS = [
    f"band_{metric}_{start}_{end}"
    for start, end in LAG_BANDS
    for metric in BAND_METRICS
]

FEATURES_PAIR = [
    "yoy_ratio",
    "pair_mean", "pair_median", "pair_max", "pair_min", "pair_count",
    "pair_std", "pair_cv", "pair_zero_rate_expanding",
    "n_active_weeks", "pair_nonzero_mean", "pair_nonzero_share",
    "pair_last_nonzero_gap",
    "pair_mean_lag26", "pair_median_lag26", "pair_max_lag26",
    "pair_min_lag26", "pair_std_lag26", "pair_cv_lag26",
    "pair_zero_rate_lag26", "pair_active_weeks_lag26",
    "pair_nonzero_mean_lag26", "pair_nonzero_share_lag26",
    "ratio_lag52_vs_pair_mean_lag26",
    "ratio_band26_52_vs_pair_mean_lag26",
    "trend_band_26_52_vs_52_104",
]

FEATURES_SEASONAL = [
    "sem_mean", "sem_max", "sem_min", "sem_median", "sem_std",
    "sem_count", "sem_zero_rate", "sem_nonzero_mean",
    "sem_lag_y1", "sem_lag_y2",
    "sem_yoy_ratio_1", "sem_yoy_ratio_2", "sem_trend_yoy",
]

FEATURES_AGENCY_ARTICLE = [
    "agence_mean", "agence_median", "article_mean", "article_median",
    "agence_mean_lag26", "agence_zero_rate_lag26", "agence_nonzero_mean_lag26",
    "article_mean_lag26", "article_zero_rate_lag26", "article_nonzero_mean_lag26",
]

FEATURES_CALENDAR = [
    "annee", "num_sem", "trimestre", "mois_approx", "week_of_quarter",
    "sin_sem", "cos_sem", "sin_sem_2", "cos_sem_2",
    "sin_sem_3", "cos_sem_3", "sin_sem_4", "cos_sem_4",
    "is_summer_trough", "is_xmas_trough", "is_august", "is_december",
    "is_quarter_end", "is_peak_spring", "is_peak_autumn",
    "weeks_to_summer", "weeks_to_xmas",
]

FEATURES_STATIC_NUMERIC = [
    "art_poids_kg", "art_log_poids_kg", "art_mdd_num",
    "ag_latitude", "ag_longitude",
]

FEATURES_BILLING = [
    "fac_prix_unit", "fac_pct_pro", "fac_pct_particulier", "fac_pct_magasin",
    "fac_nb_chantiers", "fac_nb_achats",
    "fac_sum_quantite", "fac_sum_montant",
    "fac_min_quantite", "fac_max_quantite",
    "fac_qty_per_achat", "fac_montant_per_achat",
    "fac_chantiers_per_achat", "fac_montant_per_chantier",
    "fac_qty_per_chantier",
    "fac_log_sum_quantite", "fac_log_sum_montant",
    "fac_qty_roll3", "fac_qty_roll6",
    "fac_montant_roll3", "fac_montant_roll6",
    "fac_achats_roll6", "fac_prix_unit_roll6", "fac_pct_pro_roll6",
]

FEATURES_NUMERIC = (
    FEATURES_LAGS
    + FEATURES_ROLLING
    + FEATURES_LAG_BANDS
    + FEATURES_PAIR
    + FEATURES_SEASONAL
    + FEATURES_AGENCY_ARTICLE
    + FEATURES_CALENDAR
    + FEATURES_STATIC_NUMERIC
    + FEATURES_BILLING
)

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
]

FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
