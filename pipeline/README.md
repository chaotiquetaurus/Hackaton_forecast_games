# SGDB 2026 Forecast Pipeline

Databricks-native pipeline for the Saint-Gobain Distribution × Automatants
2026 hackathon. Predicts weekly sales per (`code_agence`, `code_article`)
pair for 2025-W27 .. 2025-W52. Metric: **WAPE**.

## Architecture

```
┌──────────────────────────── LAKEFLOW (Delta Live Tables) ────────────────────────┐
│  pipeline_lakeflow.py                                                            │
│                                                                                  │
│  BRONZE (raw passthrough)                                                        │
│  ├─ bronze_ventes          ← histo_ventes_train                                  │
│  ├─ bronze_ventes_test     ← histo_ventes_test                                   │
│  ├─ bronze_agences         ← donnees_agence                                      │
│  ├─ bronze_articles        ← donnees_articles                                    │
│  └─ bronze_facturation     ← donnees_facturation                                 │
│                                                                                  │
│  SILVER (cleaned + enriched + DQ gates)                                          │
│  ├─ silver_ventes              ← cap / anomaly / smooth / dead-pair flag         │
│  │                               with @dlt.expect_or_drop on null keys          │
│  ├─ silver_articles_encoded    ← label-encoded specialite/famille/marque/mdd    │
│  ├─ silver_agences_encoded     ← label-encoded region                            │
│  ├─ silver_facturation_lagged  ← monthly aggregate, +2-month calendar lag        │
│  └─ silver_panel               ← train ∪ test (null target) + propagated         │
│                                  is_dead_pair                                    │
│                                                                                  │
│  GOLD (ML-ready)                                                                 │
│  └─ gold_feature_table         ← lags, rolling stats, pair/agency/article        │
│     │                            expanding stats, seasonal stats, billing        │
│     │                            joins, sin/cos, calendar flags, categoricals    │
│     ├─ gold_train              (<= 2024-W26)                                     │
│     ├─ gold_validation         (2024-W27 .. 2024-W52)                            │
│     ├─ gold_internal_test      (2025-W01 .. 2025-W26)                            │
│     └─ gold_test_features      (final inference: 2025-W27 .. 2025-W52)           │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────── ML NOTEBOOKS (MLflow) ───────────────────────────────┐
│  03_train_model.py   ← gold_train + gold_validation                              │
│                         → two-stage LightGBM in Model Registry, tuned threshold  │
│                                                                                  │
│  04_evaluate.py      ← val_predictions + silver_ventes (for baselines)           │
│                         → WAPE metrics + slice reports in MLflow                 │
│                                                                                  │
│  05_inference.py     ← gold_test_features + Model Registry                       │
│                         → predictions_final Delta + submission table + CSV       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## File layout

| File | Role |
|------|------|
| `00_config.py` | Single source of truth: table names, split boundaries, feature list, hyperparameters. `%run` from every notebook. |
| `pipeline_lakeflow.py` | **The entire data side** as `@dlt.table` functions. Point your Lakeflow Pipeline at this file. |
| `03_train_model.py` | Two-stage LightGBM training, MLflow logging, model registration. |
| `04_evaluate.py` | Validation WAPE + baselines + sliced diagnostics. |
| `05_inference.py` | Test-period scoring, dead-pair shortcut, CSV export. |
| `06_orchestrate.py` | Sanity-check the gold tables exist, then `%run` 03 → 04 → 05. |
| `workflow.json` | Databricks Job definition: DLT pipeline task + three notebook tasks. |
| `src/utils.py` | Shared Python helpers (WAPE, window builders, label encoders). |

## Lakeflow execution model

1. **Create a Pipeline in Databricks → Pipelines → Create pipeline.**
2. **Source:** point it at `pipeline/pipeline_lakeflow.py` (Workspace file).
3. **Target catalog / schema:** `workspace.default`.
4. **Run** → DLT resolves the DAG from the `dlt.read(...)` references and
   materialises each `@dlt.table` as a Delta table in `workspace.default`.
5. The UI shows per-expectation pass/fail rates for every silver table.

The DLT pipeline has **no side effects** — no MLflow, no prints, no
writes outside the DLT-managed target. All operational logging happens in
the ML notebooks downstream.

## Data quality expectations (silver layer)

| Table | Expectation | Action |
|-------|-------------|--------|
| `silver_ventes` | `semaine IS NOT NULL` | drop row |
| `silver_ventes` | `code_agence IS NOT NULL` | drop row |
| `silver_ventes` | `code_article IS NOT NULL` | drop row |
| `silver_ventes` | `quantite >= 0` | warn |
| `silver_ventes` | `quantite <= 25000` | warn |
| `silver_ventes` | `annee BETWEEN 2020 AND 2026` | warn |
| `silver_articles_encoded` | pair keys not null | **fail pipeline** |
| `silver_panel` | `semaine IS NOT NULL` | drop row |
| `silver_panel` | keys not null | drop row |
| `gold_train` / `gold_validation` / `gold_internal_test` | `quantite IS NOT NULL` | drop row |
| `gold_test_features` | `quantite IS NULL` | **fail pipeline** |

The last one is deliberately strict: if any test row somehow has a
non-null target the pipeline must fail, because that would mean the
train/test boundary has leaked.

## Model design

- **Two-stage hurdle model:**
  - *Stage 1* — binary LightGBM classifies `P(quantite = 0)`.
  - *Stage 2* — LightGBM regressor on raw `quantite`, trained **only on
    rows where `quantite > 0`** with a WAPE-aligned MAE objective.
- **Combining rule:** `pred = 0 if p_zero > threshold else reg_pred`.
  The threshold and a small seasonal blend are tuned on blind iterative
  validation WAPE and persisted in MLflow.
- **Dead pairs** (74 pairs with 100% zero history) are short-circuited to 0
  at inference — the `is_dead_pair` flag is propagated from silver to gold
  in DLT so inference needs no extra lookup.
- **Lag handling at validation/test/inference:** horizons are scored
  recursively. After each predicted week, the prediction is fed back into the
  in-memory history and the next week's lags, rolling stats, zero/activity
  rates and recency features are rebuilt from that history.

## Temporal splits

| Set | `semaine` range |
|-----|-----------------|
| Train | `<= 2024-W26` |
| Validation | `2024-W27 .. 2024-W52` |
| Internal labelled test | `2025-W01 .. 2025-W26` |
| Final inference / submission | `2025-W27 .. 2025-W52` |

The DLT `gold_*` tables encode these filters.

## Leakage guarantees

- Every window in `gold_feature_table` uses `rowsBetween(..., -1)` — the
  current row is always excluded.
- Billing features are lagged by 2 calendar months inside
  `silver_facturation_lagged` before the join, so the model never sees
  the monthly invoicing summary that contains the week it predicts.
- Seasonal stats (`sem_*`) partition on `(pair, num_sem)` and order by
  year, so they see only past same-week values.
- Agency and article expanding stats ignore rows where `y` is null
  (F.avg/stddev skip nulls), so test rows never contribute to them.

## Run on Databricks

### Option A — Interactive

1. Import `pipeline/` to your workspace.
2. **Data side:** open Pipelines UI → create pipeline → source = `pipeline_lakeflow.py` → run.
3. **ML side:** open `06_orchestrate.py` on a cluster with `lightgbm==4.3.0` and click Run All.

### Option B — Scheduled Job

```bash
databricks jobs create --json-file pipeline/workflow.json
```

Replace `<REPLACE_WITH_DLT_PIPELINE_ID>` in `workflow.json` with the
pipeline ID from the Lakeflow UI before creating the job.

## Submission

`05_inference.py` writes the final predictions to:

- Delta: `workspace.default.predictions_final`
- Submission table: `workspace.default.predictions_equipe_telecacaton`
- CSV: `/dbfs/FileStore/sgdb2026_submission.csv`

Submit via the hackathon app:
<https://hackathon-sgdb-leaderboard-7474650176048310.aws.databricksapps.com>

## Baselines to beat

| Baseline | WAPE (val) |
|----------|------------|
| Naive lag-52 | ~1.387 |
| Blend (0.6·lag52 + 0.4·pair_mean) | ~1.259 |
| Two-stage LightGBM (this pipeline) | *computed at run time in 04_evaluate* |
