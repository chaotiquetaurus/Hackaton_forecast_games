# Plan de tests — J+1 après sélection de features LightGBM

Contexte :
- Un **LightGBM propre** est déjà disponible.
- Les **features ont déjà été sélectionnées** parmi ~250 variables.
- L'objectif de demain est de **tester des axes d'amélioration de la prédiction**.
- On se concentre sur les **expériences à lancer**, pas sur le nettoyage de données.

---

## 1. Data Analysis

### 1.1 Benchmark propre des features retenues
À faire :
- lister les **top features** finales
- mesurer leur **importance gain**
- calculer une **permutation importance** sur validation temporelle
- analyser les features par **blocs** : lags, rolling stats, intermittence, calendrier, agrégations, facturation

Mots-clés :
- `gain importance`
- `permutation importance`
- `feature blocks`
- `validation temporelle`
- `ablation study`

### 1.2 Analyse d'erreurs
À faire :
- regarder les erreurs sur les **grosses quantités**
- regarder les erreurs sur les **séries très intermittentes**
- regarder les erreurs par **agence**
- regarder les erreurs par **famille / sous-famille**
- séparer les cas : `vente > 0` vs `vente = 0`

Mots-clés :
- `error slicing`
- `sparse series`
- `heavy tail`
- `per-agency error`
- `per-family error`
- `zero vs non-zero`

### 1.3 Analyse des résidus
À faire :
- vérifier si le modèle **sous-prédit les pics**
- vérifier si le modèle **sur-prédit les séries mortes**
- regarder la distribution des résidus par segment
- comparer résidus sur `quantite faible`, `quantite moyenne`, `quantite forte`

Mots-clés :
- `residual analysis`
- `underprediction of peaks`
- `overprediction on zeros`
- `segment diagnostics`

---

## 2. Data Science

### 2.1 Tuning du LightGBM final
À tester :
- `learning_rate`
- `n_estimators`
- `num_leaves`
- `max_depth`
- `min_data_in_leaf`
- `feature_fraction`
- `bagging_fraction`
- `lambda_l1`
- `lambda_l2`

Objectif :
- tirer le maximum du modèle **après sélection des features**

Mots-clés :
- `hyperparameter tuning`
- `regularization`
- `leaf-wise growth`
- `temporal CV`

### 2.2 Tester la cible brute vs cible transformée
À tester :
- `y = quantite`
- `y = log1p(quantite)` puis retour avec `expm1`

Objectif :
- voir si la transformation réduit l'effet des grosses valeurs
- stabiliser la prédiction sur les distributions asymétriques

Mots-clés :
- `target transform`
- `log1p target`
- `heavy-tailed target`

### 2.3 Modèle 2 étages : occurrence + quantité
À tester :
- **modèle 1** : classification `vente > 0`
- **modèle 2** : régression sur les cas `quantite > 0`
- combinaison finale : `proba_vente * quantite_conditionnelle`

Objectif :
- mieux gérer les **67 % de zéros**
- séparer le problème : `est-ce qu'on vend ?` puis `combien si on vend ?`

Mots-clés :
- `two-stage model`
- `hurdle model`
- `occurrence model`
- `conditional demand`
- `zero inflation`

### 2.4 CatBoost comme vrai challenger
À tester :
- entraîner un **CatBoost** sur les mêmes folds temporels
- utiliser les mêmes features finales
- comparer directement au LightGBM

Pourquoi :
- beaucoup de **catégorielles** dans votre dataset
- CatBoost peut être très bon sur ce type de problème

Mots-clés :
- `CatBoost`
- `categorical features`
- `global tabular forecasting`

### 2.5 XGBoost comme second challenger
À tester :
- entraîner un **XGBoost** sur le même protocole
- soit avec toutes les features finales
- soit avec un sous-ensemble très robuste

Objectif :
- obtenir un modèle concurrent
- préparer un futur ensemble

Mots-clés :
- `XGBoost`
- `booster comparison`
- `robust subset`

### 2.6 Ensemble de modèles
À tester :
- `LightGBM + CatBoost`
- `LightGBM + XGBoost`
- moyenne simple `50/50`
- moyenne pondérée avec recherche du meilleur poids

Forme :
- `pred_mix = w * pred_model_1 + (1 - w) * pred_model_2`

Objectif :
- profiter d'erreurs différentes entre modèles

Mots-clés :
- `ensemble`
- `weighted average`
- `OOF predictions`
- `stacking lite`

### 2.7 Baselines spécialisées demande intermittente
À tester :
- `Croston`
- `TSB`
- `IMAPA`

Objectif :
- comparer votre pipeline ML à des méthodes spécialisées sur séries très creuses
- savoir si votre approche tabulaire bat bien les baselines intermittentes

Mots-clés :
- `intermittent demand`
- `Croston`
- `TSB`
- `IMAPA`

### 2.8 AutoML / benchmark large
À tester si temps disponible :
- `AutoGluon TimeSeries`

Objectif :
- benchmark rapide sur un panel de modèles
- voir si un ensemble AutoML surperforme les boosters manuels

Mots-clés :
- `AutoML`
- `AutoGluon TimeSeries`
- `benchmark`

---

## 3. Ordre recommandé pour demain

### Priorité 1
- benchmark propre du **LightGBM final**
- **tuning** du LightGBM
- test `quantite` vs `log1p(quantite)`

### Priorité 2
- **modèle 2 étages** : occurrence + quantité
- **CatBoost** sur les mêmes features

### Priorité 3
- **ensemble** LightGBM + CatBoost
- **XGBoost** comme challenger additionnel

### Priorité 4
- baselines intermittentes : **Croston / TSB / IMAPA**
- **AutoGluon TimeSeries** si temps restant

---

## 4. Questions auxquelles répondre demain

- Le **tuning** améliore-t-il significativement le LightGBM final ?
- La cible `log1p` est-elle meilleure que la cible brute ?
- Le **modèle 2 étages** gère-t-il mieux les nombreux zéros ?
- **CatBoost** bat-il LightGBM sur vos features finales ?
- Un **ensemble** améliore-t-il la métrique ?
- Vos boosters battent-ils clairement les méthodes intermittentes classiques ?

---

## 5. Sorties à produire à la fin de la journée

- tableau comparatif des modèles testés
- métrique par modèle
- métrique par segment : `zéros`, `non-zéros`, `grosses ventes`
- meilleur modèle seul
- meilleur ensemble
- décision claire sur la suite

Mots-clés finaux :
- `benchmark`
- `temporal CV`
- `model comparison`
- `two-stage forecasting`
- `ensemble`
- `intermittent demand`
