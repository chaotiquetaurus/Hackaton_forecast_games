# -----------------------------------------------------------------------------
# Configuration — Single source of truth for the SGDB 2026 DLT pipeline
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
ANOMALY_ROLL_WINDOW = 26

# Anomaly detection: MAD-based z-score.
ANOMALY_MAD_ZSCORE = 3.5

# Feature engineering knobs
LAGS_ALL = [1, 2, 4, 8, 13, 26, 27, 52, 104]

ROLLING_WINDOWS = [4, 8, 13, 26, 52]
ROLLING_MEDIAN_WINDOWS = [4, 13, 26, 52]

# Band windows — (lo, hi) means rows -hi .. -lo (past only, inference-safe).
BAND_WINDOWS = [
    (26, 52), (27, 52), (39, 65), (52, 78), (52, 104), (78, 104), (104, 156),
]
