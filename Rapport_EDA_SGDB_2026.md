# Analyse Exploratoire — Hackathon SGDB France 2026

**Objectif :** Prédire les quantités de ventes hebdomadaires par paire agence × article sur S27–S52 2025.  
**Métrique :** WAPE (Weighted Absolute Percentage Error) — plus c'est bas, mieux c'est.  
**Baselines à battre :** Saisonnière N-1 → 1.387 | Blend pondéré → 1.259

---

## 1. Structure des données

### 1.1 Tables disponibles

| Table | Lignes | Colonnes | Rôle |
|-------|--------|----------|------|
| histo_ventes_train | 2 354 189 | 4 | Historique de ventes (semaine, agence, article, quantité) |
| histo_ventes_test | 272 344 | 3 | Lignes à prédire (quantité manquante) |
| donnees_agence | 14 | 9 | Référentiel agences (région, GPS, secteur…) |
| donnees_articles | 11 908 | 11 | Référentiel articles **par paire agence×article** |
| donnees_facturation | 358 999 | 16 | Données de facturation mensuelles par paire |

### 1.2 Aucune valeur manquante

Toutes les tables ont 0% de nulls sur toutes les colonnes, à l'exception de `code_fournisseur` dans donnees_articles (3 nulls sur 11 908, négligeable).

### 1.3 Périmètre temporel

- **Train :** 2021-01 → 2025-26 (234 semaines, soit ~4.5 ans)
- **Test :** 2025-27 → 2025-52 (26 semaines, second semestre 2025)
- **Validation recommandée :** S01–S26 2025 (premier semestre 2025)

---

## 2. Distribution de la variable cible (quantite)

| Statistique | Valeur |
|-------------|--------|
| **% de zéros** | **67.1%** |
| Médiane | 0 |
| Moyenne | 10.48 |
| Écart-type | 113.16 |
| P75 | 1 |
| P90 | 8 |
| P95 | 24 |
| P99 | 179 |
| Max | 21 646 |

**Constats clés :**
- Distribution extrêmement asymétrique : deux tiers des observations sont des zéros.
- La moyenne (10.5) est très éloignée de la médiane (0) → distribution dominée par quelques grosses valeurs.
- L'écart-type (113) est 10× la moyenne → très forte dispersion.
- Le P99 à 179 vs un max à 21 646 indique des valeurs extrêmes rares mais massives.

---

## 3. Analyse temporelle

### 3.1 Tendance annuelle : baisse continue

| Année | Avg quantité min | Avg quantité max | Avg quantité moyenne |
|-------|-------------------|-------------------|----------------------|
| 2021 | 4.54 | 16.07 | **12.04** |
| 2022 | 5.45 | 16.84 | **12.22** |
| 2023 | 3.57 | 14.58 | **10.26** |
| 2024 | 2.71 | 12.44 | **8.64** |
| 2025 | 2.57 | 12.29 | **8.56** |

La tendance est nettement baissière : -30% entre 2021-2022 et 2024-2025. Le modèle doit capter cette décroissance sous peine de surestimer systématiquement.

### 3.2 Saisonnalité intra-annuelle

**Pics (semaines fortes) :** S11 (12.3), S13 (12.1), S24-S25 (12.5-13.0), S36 (12.1), S38 (12.6), S40 (13.1), S46 (12.8)

**Creux (semaines faibles) :** S33 (4.6), S52 (4.6), S01 (8.0), S32 (6.7), S51 (7.9)

Le profil saisonnier montre deux creux marqués : les vacances d'été (S32-S33, août) et les fêtes de fin d'année (S51-S52). Le test couvre S27-S52, donc il inclut ces deux creux — le modèle doit bien les capter.

### 3.3 Semaines extrêmes

**Top volume :** S19/2022 (168K), S13/2022 (159K), S42/2022 (158K) — 2022 domine le top 10.

**Bottom volume :** S01/2025 (27K), S52/2024 (28K), S52/2023 (36K) — les semaines de fin/début d'année.

---

## 4. Analyse des agences (14 agences)

### 4.1 Répartition du volume

| Agence | Volume total | Avg qty | Nb articles | % zéros |
|--------|-------------|---------|-------------|---------|
| 3168 (Ludres, Est) | 4 317 962 | 13.4 | 1 644 | 67.0% |
| 1527 (Sucy-en-Brie, IDF) | 3 579 943 | 9.3 | 1 812 | 65.5% |
| 4565 (Créon, Sud-Ouest) | 3 385 440 | 11.8 | 1 466 | 66.0% |
| 3536 (Montbert, Pays de Loire) | 3 141 672 | 16.4 | 996 | 70.6% |
| 4104 (Bruguières, Méridionale) | 2 254 635 | 19.9 | 621 | 69.2% |
| … | … | … | … | … |
| 5480 (Feytiat, Sud-Ouest) | 232 478 | 4.6 | 319 | 77.4% |
| 1951 (Annezin, Nord) | 104 656 | 3.3 | 183 | 66.2% |
| 1343 (Mâcon, Rhône-Alpes) | 78 176 | 2.4 | 169 | 68.4% |

**Constats :**
- Forte hétérogénéité : l'agence 3168 fait 55× le volume de l'agence 1343.
- Toutes les agences ont >50% de zéros (aucune <50%).
- L'agence 4104 a le meilleur avg (19.9) mais seulement 621 articles → spécialisée et dense.
- L'agence 1527 a le plus d'articles (1 812) mais un avg modéré (9.3) → large catalogue, beaucoup de faibles volumes.

### 4.2 Répartition géographique

| Région | Volume | Nb agences |
|--------|--------|------------|
| Est | 4 908 396 | 2 |
| Méridionale | 3 628 307 | 2 |
| Sud-Ouest | 3 617 918 | 2 |
| IDF | 3 579 943 | 1 |
| Pays de Loire | 3 141 672 | 1 |
| Nord | 1 870 967 | 2 |
| Normandie | 1 869 187 | 1 |
| Centre | 1 243 677 | 1 |
| PACA | 745 171 | 1 |
| Rhône-Alpes | 78 176 | 1 |

Toutes les agences sont du métier "Négoce" → cette colonne est constante et inutile comme feature.

---

## 5. Analyse des articles (5 789 articles)

### 5.1 Concentration extrême (Pareto)

**441 articles (7.6%) représentent 80% du volume total.**

- 250 articles ont >90% de zéros.
- Le top article (6041153) fait 739K de volume avec seulement 3 agences et 51% de zéros → gros volumes concentrés.
- Certains articles (6271640, 4393730) ont <1% de zéros → ventes très régulières.

### 5.2 Enrichissement par catégorie

**Spécialité (17 valeurs) :**

| Spécialité | Volume | Nb articles |
|-----------|--------|-------------|
| Gros-œuvre | 22.6M | 1 186 |
| Plafond/Plâtrerie/Isolation | 12.2M | 761 |
| Couverture | 12.1M | 1 186 |
| Bois de construction | 3.0M | 311 |
| Revêtements | 2.7M | 370 |

Les 3 premières spécialités couvrent ~75% du volume.

**Famille (162 valeurs) — Top 5 :**

| Famille | Volume | Nb articles |
|---------|--------|-------------|
| Tuiles et accessoires | 8.3M | 584 |
| Ossatures plaques de plâtre | 5.7M | 139 |
| Briques de structure terre cuite | 4.4M | 66 |
| Blocs | 4.4M | 194 |
| Ronds à béton | 4.1M | 8 |

**Marque :** La marque "-" (sans marque) et "ULTIBAT" dominent avec ~7.5M et ~6.9M respectivement.

**MDD :** Les articles non-MDD (article_mdd=0) représentent 79.5% du volume avec un avg de 7.3, vs 3.5 pour les MDD.

**Unité de vente :** "Pièce" domine en nb d'articles (2 940) mais "Milliers" a le plus haut avg (291.8).

### 5.3 Point critique — donnees_articles est indexée par (agence, article)

La table articles a 11 908 lignes pour 5 789 articles × 14 agences. Elle contient `code_agence` en plus de `code_article`. Cela signifie que les caractéristiques d'un article peuvent varier selon l'agence (ex : gamme, sous-famille). La jointure doit se faire sur les deux clés.

---

## 6. Paires agence × article

### 6.1 Couverture test/train

| Métrique | Valeur |
|----------|--------|
| Paires uniques train | 11 908 |
| Paires uniques test | 11 690 |
| **Paires test absentes du train** | **0 (0%)** |
| Agences test absentes du train | 0 |
| Articles test absents du train | 0 |

**Aucun cold-start.** Toutes les paires du test existent dans le train. C'est une très bonne nouvelle : on peut s'appuyer entièrement sur l'historique.

### 6.2 Distribution des paires

| Métrique | Valeur |
|----------|--------|
| Paires 100% zéros | 74 |
| Paires >90% zéros | 521 |
| Paires avg > 10 | 1 316 |
| Paires avg > 100 | 213 |
| Paires 1 seule semaine | 0 |

Toutes les paires couvrent les 234 semaines de train → pas de paires éphémères. 521 paires (4.4%) sont quasi-mortes (>90% zéros).

---

## 7. Facturation (données mensuelles)

### 7.1 Structure

Granularité : **agence × article × année × mois** (358 999 lignes).

Colonnes clés exploitables :

| Colonne | Description | Utilité |
|---------|-------------|---------|
| nb_achats | Nombre de transactions | Proxy d'activité |
| sum_quantite / sum_montant | Volume et CA mensuels | Prix unitaire dérivable |
| nb_achats_par_professionnels | Part pro | Segmentation client |
| nb_achats_par_particuliers | Part particulier | Segmentation client |
| nb_ventes_magasins | Ventes en magasin | Canal de vente |
| nb_ventes_directes | Ventes directes | Canal (quasi nul : avg 0.01) |
| nb_chantiers | Nb chantiers distincts | Diversification demande |

### 7.2 Stats descriptives

- **nb_achats** : médiane 2, moyenne 3.95, max 244 → distribution très asymétrique.
- **sum_quantite** : médiane 5, moyenne 53.3, max 15 969.
- **sum_montant** : médiane 72.6€, moyenne 389€, max 44 121€.
- **nb_ventes_directes** : médiane 0, moyenne 0.01 → quasi-inexistant, feature inutile.
- **Prix unitaire moyen** dérivable via `sum_montant / sum_quantite`.

---

## 8. Corrélations des lags

| Lag | Corrélation avec quantite | Nb observations |
|-----|--------------------------|-----------------|
| Lag-1 | 0.3173 | 2 342 281 |
| Lag-4 | 0.2961 | 2 306 557 |
| Lag-12 | 0.2886 | 2 211 370 |
| Lag-26 | 0.2882 | 2 045 472 |
| Lag-52 | 0.2998 | 1 741 735 |

**Constat surprenant :** les corrélations sont toutes modérées (~0.29-0.32) et relativement proches. Le lag-52 (même semaine année précédente) n'est pas significativement plus fort que le lag-1. Cela s'explique par :

1. La forte proportion de zéros (67%) qui écrase les corrélations.
2. La tendance baissière qui réduit la valeur prédictive du N-1.
3. Sur les paires actives (non-zéros), la corrélation serait probablement bien plus forte.

Le lag-1 a la plus forte corrélation (0.317), mais il ne sera pas disponible en test puisqu'on prédit 26 semaines d'un coup. Seul le lag-52 est directement utilisable pour toutes les semaines test.

---

## 9. Synthèse des features

### 9.1 Features prioritaires

| Feature | Catégorie | Priorité | Risque leakage |
|---------|-----------|----------|----------------|
| qty_lag52 | Lag | ★★★★★ | NaN si paire nouvelle (0 cas ici) |
| pair_expanding_mean | Stats paire | ★★★★★ | Calculer sur données < S |
| rolling_mean_4w | Rolling | ★★★★ | SHIFT obligatoire |
| rolling_mean_12w | Rolling | ★★★★ | SHIFT obligatoire |
| zero_rate_4w / 12w | Zéros | ★★★★ | SHIFT obligatoire |
| art_famille | Enrichissement | ★★★★ | Aucun |
| art_specialite | Enrichissement | ★★★ | Aucun |
| ag_region | Enrichissement | ★★★ | Aucun |
| sem_sin / sem_cos | Saisonnalité | ★★★ | Aucun |
| fac_prix_unit | Facturation | ★★★ | Jointure à vérifier |

### 9.2 Features dangereuses (leakage)

| Danger | Feature concernée | Règle |
|--------|-------------------|-------|
| 🔴 Fuite temporelle | Toute rolling/expanding | Fenêtre `rowsBetween(..., -1)`, jamais inclure la ligne courante |
| 🔴 Lag indisponible | lag_1 à lag_25 | En test on prédit S27-S52 d'un bloc, ces lags n'existent pas |
| 🔴 Stats globales | mean/max sur toute la série | Doivent être expanding (uniquement données passées) |
| ⚠️ Facturation mensuelle | fac_* | Granularité mois ≠ semaine, agréger par paire sans futur |
| ⚠️ Target encoding | Encoder catégorielles par la moyenne cible | Utiliser fold-based ou calculer sur train uniquement |
| ✅ Safe | num_semaine, sin/cos, région, famille, marque, MDD | Pas de risque |

### 9.3 Feature inutile identifiée

- **metier** : constante ("Négoce" pour les 14 agences) → aucun pouvoir discriminant.
- **nb_ventes_directes** : médiane 0, moyenne 0.01 → quasi-constante.

---

## 10. Recommandations stratégiques

1. **Le zéro est roi.** 67% de la cible est à zéro. Un modèle qui prédit bien les zéros a un avantage massif. Envisager un modèle en deux étapes : classifier zéro/non-zéro puis régresser la quantité si non-zéro.

2. **Capter la tendance baissière.** Le volume moyen a chuté de 30% entre 2021 et 2025. Les features de tendance (ratio court/long terme, YoY) sont essentielles.

3. **Le lag-52 est nécessaire mais pas suffisant.** Corrélation modeste (0.30). Le combiner avec des moyennes mobiles longues et des stats paire donne de meilleurs résultats qu'un simple copy N-1.

4. **Exploiter la facturation.** Le prix unitaire, le % pro vs particulier, et le nb de chantiers apportent une information orthogonale aux séries temporelles pures.

5. **Le test couvre l'été + Noël.** Les deux creux saisonniers les plus forts (S32-33 et S51-52) sont dans la période test. Le modèle doit absolument capter ce profil.

6. **Blend de modèles.** Le guide mentionne que "un blend bat généralement un modèle seul". Combiner LightGBM + XGBoost + baseline saisonnière pondérée est la stratégie optimale.
