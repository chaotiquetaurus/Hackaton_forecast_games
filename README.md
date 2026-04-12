# Hackathon Automatants x Saint-Gobain x Margo — Sales Forecasting on Databricks

> **April 10–12, 2026** | Organized by [Margo](https://www.margo-group.com/), [Saint-Gobain Distribution France (SGDBF)](https://www.saint-gobain.com/), [Databricks](https://www.databricks.com/), and the student association [Automatants](https://automatants.music-all.com/).

## Objective

Build a **full end-to-end pipeline** for forecasting **weekly sales quantities** for every (agency × product) pair over the second half of 2025 (weeks 27–52).

The goal is to minimize the **WAPE** (Weighted Absolute Percentage Error):

$$
WAPE = \frac{\sum_{(week,\, agency,\, article)} |\ quantity_{actual} - quantity_{predicted}\ |}{\sum_{(week,\, agency,\, article)}\ quantity_{actual}}
$$

Lower is better. Provided baselines to beat: **Seasonal N−1 → 1.387** | **Weighted blend → 1.259**.

---

## Why Databricks?

Everything in this project lives on **Databricks**. The platform provides:

- **Data storage** — All datasets are hosted in the Unity Catalog (`workspace.default`): `histo_ventes_train`, `histo_ventes_test`, `donnees_agence`, `donnees_articles`, `donnees_facturation`.
- **Compute** — Serverless clusters for scalable Spark and Python workloads.
- **Native notebooks** — Interactive exploration, PySpark, Pandas, SQL, and visualization in one place.
- **Data engineering** — Delta Live Tables, Jobs & Pipelines for orchestrating feature engineering and model training.
- **MLflow** — Experiment tracking, model registry, and lifecycle management, all built-in.
- **Catalog Explorer** — Browse tables, inspect schemas, preview sample data, check lineage and quality.

**We aim to use Databricks-native tools wherever possible** — PySpark for feature engineering, MLflow for experiment tracking, Delta tables for intermediate storage, and Jobs for pipeline orchestration — rather than running everything locally or in ad-hoc scripts.

---

## Data Split & Validation Strategy

The temporal split is **fixed and strict** — this is critical for avoiding data leakage:

| Split | Period | Weeks |
|---|---|---|
| **Train** | 2021-W01 → 2025-W26 | 234 weeks (~4.5 years) |
| **Validation** | 2025-W01 → 2025-W26 | 26 weeks (H1 2025, carved from train) |
| **Test** | 2025-W27 → 2025-W52 | 26 weeks (H2 2025, quantity hidden) |

All lag features, rolling windows, and expanding statistics **must be computed strictly on past data** (no leakage from the current or future rows). Since we predict 26 weeks at once, lag-1 through lag-25 are **not available** at inference time — only lag-52+ can be used directly.

---

## Evaluation Criteria

### Pre-selection (code submission)
- **WAPE score** on the hidden test set
- **Code quality**: reproducibility, time-series rigor (no leakage), structure & modularity, readability, documentation

### Final selection (oral presentation)
- Understanding of the business problem
- Time-series methodology
- Modeling choices and justification
- Clarity of presentation and supporting material
- Q&A

> **A simple, well-built model is valued more than a complex, poorly-mastered one.**

---

## Data Overview

### Available Tables

| Table | Rows | Columns | Description |
|---|---|---|---|
| `histo_ventes_train` | 2,354,189 | 4 | Weekly sales history (week, agency, article, quantity) |
| `histo_ventes_test` | 272,344 | 3 | Rows to predict (quantity missing) |
| `donnees_agence` | 14 | 9 | Agency reference data (region, GPS, sector…) |
| `donnees_articles` | 11,908 | 11 | Product reference data **keyed by (agency, article)** |
| `donnees_facturation` | 358,999 | 16 | Monthly billing data per (agency, article) pair |

No missing values across all tables (except 3 nulls in `code_fournisseur`, negligible). **Zero cold-start**: every (agency × article) pair in the test set exists in the training set.

### Target Variable — `quantite`

The target is highly skewed: **67% of observations are zero**, the mean is 10.5 but the median is 0, and the standard deviation (113) is 10× the mean. A few extreme values reach 21,646. This distribution demands careful handling — a two-stage model (zero/non-zero classifier + quantity regressor) is a strong approach.

### Key Patterns

- **Downward trend**: average weekly quantity dropped ~30% between 2021 and 2025. The model must capture this decline to avoid systematic over-prediction.
- **Seasonality**: two deep troughs in the test period — summer holidays (W32–33) and end-of-year holidays (W51–52). Peak weeks around W24–25 and W38–40.
- **Agency heterogeneity**: 14 agencies across France with vastly different volumes (55× ratio between largest and smallest). All agencies are in the "Négoce" (wholesale) sector.
- **Pareto concentration**: 441 articles (7.6%) account for 80% of total volume. Top product families: tiles & accessories, plasterboard frames, structural bricks.
- **Lag correlations**: all lags show moderate correlation (~0.29–0.32) with the target, dampened by the high zero-rate. Lag-52 (same week last year) is the most directly usable at inference time.

### Enrichment Sources

- **Article metadata**: specialty (17 values), family (162 values), brand, private-label flag, unit of sale — joined on **(agency, article)**, not article alone.
- **Billing data**: monthly granularity with transaction count, volume, revenue, professional/retail split, number of construction sites. Derivable: unit price, channel mix.

---

## Recommended Approach

1. **Feature engineering on Databricks** (PySpark / Delta) — lag-52, expanding means, rolling statistics (shifted), seasonal encodings (sin/cos of week number), year-over-year ratios, article and agency categorical features, billing-derived features.
2. **Train with gradient-boosted trees** — LightGBM / XGBoost, treating this as a tabular regression problem with careful temporal cross-validation.
3. **Track experiments with MLflow** — log parameters, metrics (WAPE), and models.
4. **Blend models** — a weighted ensemble of multiple models typically outperforms any single model.
5. **Orchestrate with Databricks Jobs** — chain feature engineering → training → inference into a reproducible pipeline.

---

## Repository Structure

```
├── README.md                        # This file
├── Rapport_EDA_SGDB_2026.md         # Full exploratory data analysis report
├── lakeflow/                        # Databricks DLT pipeline (feature engineering)
│   ├── config.py                    # DLT-specific configuration
│   └── transformations/
│       ├── bronze/bronze_tables.py  # Raw table passthrough
│       ├── silver/silver_tables.py  # Cleaning, enrichment, panel expansion
│       └── gold/gold_tables.py      # ML-ready feature table + train/val/test splits
├── pipeline/                        # ML notebooks (run on Databricks)
│   ├── 00_config.py                 # Single source of truth (tables, splits, hyperparams)
│   ├── 03_train_model.py            # Two-stage LightGBM (zero clf + qty regressor)
│   ├── 04_evaluate.py               # Validation WAPE, baselines, confusion matrix
│   ├── 05_inference.py              # Score test period + write submission
│   └── src/utils.py                 # Shared utilities (WAPE, windows, encoders)
```

---

## Getting Started

1. Open the Databricks workspace provided for the hackathon.
2. Navigate to **Catalog → workspace → default** to browse all five tables.
3. Start a serverless cluster and open a notebook.
4. Read `Rapport_EDA_SGDB_2026.md` for a detailed understanding of the data before writing any code.
