# House Price Prediction — Pipeline ML complet sur Snowflake

> **Évaluation Data Engineering & Machine Learning avec Snowflake**  
> **MBAESG_MBIAD_02_EVALUATION_DATAENGINEER_MLOPS**

---

## Description du projet

Ce projet implémente un pipeline complet de **Data Engineering et Machine Learning** directement dans l'environnement **Snowflake**, sans extraction de données vers des systèmes externes. Il utilise les données de caractéristiques de maisons pour prédire leur prix de vente.

**Dataset source :** `s3://logbrain-datalake/datasets/house_price/`  
**Format :** JSON  
**Nombre de lignes :** 1 090 brutes (545 uniques après déduplication)

---

## Contenu du dépôt

```
.
├── HOUSE_PRICE_PREDICTION_NOTEBOOK.ipynb   # Notebook Snowflake — pipeline ML complet
├── streamlit_app.py                        # Application Streamlit de prédiction
└── README.md                               # Ce fichier — documentation & analyse
```

---

## Dataset

| Colonne | Type | Description |
|---------|------|-------------|
| `PRICE` | Float | Prix de vente de la maison **(variable cible)** |
| `AREA` | Float | Surface totale (m²) |
| `BEDROOMS` | Int | Nombre de chambres |
| `BATHROOMS` | Int | Nombre de salles de bain |
| `STORIES` | Int | Nombre d'étages |
| `MAINROAD` | String | Accès à une route principale (yes/no) |
| `GUESTROOM` | String | Chambre d'amis (yes/no) |
| `BASEMENT` | String | Sous-sol (yes/no) |
| `HOTWATERHEATING` | String | Chauffage eau chaude (yes/no) |
| `AIRCONDITIONING` | String | Climatisation (yes/no) |
| `PARKING` | Int | Places de stationnement |
| `PREFAREA` | String | Zone privilégiée (yes/no) |
| `FURNISHINGSTATUS` | String | Ameublement (furnished / semi-furnished / unfurnished) |

---

## Architecture du pipeline

```
S3 (logbrain-datalake) — Format JSON
          │
          ▼
    External Stage
          │
          ▼
    Table HOUSE_PRICE (1090 lignes brutes)
          │
          ▼
    Déduplication (545 lignes uniques)
          │
          ▼
    EDA (Snowpark + Altair)
          │
          ▼
    Feature Engineering
    ├── Encodage binaire (yes/no → 0/1)
    ├── Encodage ordinal (FURNISHINGSTATUS)
    ├── Normalisation (StandardScaler → table SCALER_PARAMS)
    └── Split Train 80% (436) / Test 20% (109)
          │
          ├──► Régression Linéaire (baseline)
          ├──► Random Forest Regressor
          └──► XGBoost Regressor
                    │
                    ▼
            Grid Search (48 combinaisons)
            + ExperimentTracking Snowflake ML
                    │
                    ▼
            Meilleur modèle → Model Registry
                    │
                    ▼
            Inférence → HOUSE_PRICE_PREDICTIONS
                    │
                    ▼
            Application Streamlit
```

---

## Analyse des performances des modèles

### Étape 1 — Comparaison initiale des modèles (sans optimisation)

| Modèle | Train R² | Test R² | Test MAE | Test RMSE |
|--------|----------|---------|----------|-----------|
| Régression Linéaire | 0.6854 | **0.6492** | 49 000 | 66 579 |
| Random Forest | 0.9501 | 0.6125 | 51 087 | 69 977 |
| XGBoost | 0.9966 | 0.5840 | 53 305 | 72 504 |

**Observations :**
- La **Régression Linéaire** obtient le meilleur R² test (0.6492) et la MAE la plus basse, servant de baseline solide.
- **Random Forest** présente un sur-apprentissage marqué (train R² = 0.95 vs test R² = 0.61).
- **XGBoost** sans tuning souffre du sur-apprentissage le plus sévère (train R² = 0.9966 vs test R² = 0.584) sur ce petit dataset de 545 lignes.

---

### Étape 2 — Optimisation des hyperparamètres (Grid Search)

#### Random Forest — Grille testée

| Paramètre | Valeurs testées |
|-----------|----------------|
| `n_estimators` | 100, 200 |
| `max_depth` | None, 10, 20 |
| `min_samples_leaf` | 1, 3 |
| `max_features` | 'sqrt', 'log2' |

#### XGBoost — Grille testée

| Paramètre | Valeurs testées |
|-----------|----------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.05, 0.1 |
| `subsample` | 0.8, 1.0 |

**Total : 48 combinaisons testées** et trackées via `ExperimentTracking` Snowflake ML.

---

### Étape 3 — Meilleur modèle final

**Algorithme sélectionné : XGBoost Regressor (optimisé)**

| Hyperparamètre | Valeur optimale |
|----------------|----------------|
| `n_estimators` | 200 |
| `max_depth` | 3 |
| `learning_rate` | 0.1 |
| `subsample` | 1.0 |
| `random_state` | 42 |

**Performances finales :**

| Métrique | Valeur |
|----------|--------|
| **R² (test)** | **0.6636** |
| **MAE (test)** | **47 644 USD** |
| **RMSE (test)** | **65 196 USD** |

---

### Interprétation des résultats

- Un **R² de 0.6636** signifie que le modèle explique **66.4% de la variance** des prix — performance modeste liée à la taille réduite du dataset (545 lignes).
- Un **MAE de 47 644 USD** signifie qu'en moyenne, le modèle se trompe d'environ 48 000 USD sur le prix prédit.
- Le **RMSE de 65 196 USD** indique que certaines erreurs importantes persistent, notamment sur les biens à prix élevé (> 400 000 USD).
- Des features supplémentaires (localisation, année de construction, etc.) seraient nécessaires pour améliorer significativement les prédictions.

### Features les plus influentes (d'après XGBoost)

1. **AREA** — La surface est le facteur le plus déterminant du prix
2. **BATHROOMS** — Lié à la qualité et la taille du bien
3. **AIRCONDITIONING** — Indicateur de confort premium
4. **STORIES** — Nombre d'étages
5. **PREFAREA** — Zone géographique privilégiée
6. **FURNISHINGSTATUS** — Meublé > semi-meublé > non meublé
7. **PARKING** — Nombre de places de stationnement

---

## Instructions d'exécution

### Prérequis Snowflake
- Rôle avec droits `CREATE DATABASE`, `CREATE SCHEMA`, `CREATE STAGE`, `CREATE TABLE`, `CREATE MODEL`
- Accès au bucket S3 `s3://logbrain-datalake/`

### Packages Python à installer (menu Packages du notebook)
```
xgboost
snowflake-ml-python
```
> `scikit-learn`, `pandas` et `altair` sont inclus par défaut.

### Ordre d'exécution
1. Ouvrir `HOUSE_PRICE_PREDICTION_NOTEBOOK.ipynb` dans **Snowflake Notebooks**
2. Installer les packages requis
3. Exécuter les cellules dans l'ordre (étapes 1 à 10)
4. L'application Streamlit est déployée via :
   ```sql
   CREATE OR REPLACE STREAMLIT HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_APP
     ROOT_LOCATION = '@HOUSE_PRICE_DB.ML_SCHEMA.STREAMLIT_STAGE'
     MAIN_FILE = 'streamlit_app.py'
     QUERY_WAREHOUSE = 'COMPUTE_WH';
   ```

---

## Technologies utilisées

| Technologie | Usage |
|-------------|-------|
| **Snowflake Notebooks** | Environnement de développement |
| **Snowpark (Python)** | Manipulation des données dans Snowflake |
| **scikit-learn** | Préparation des données + modèles ML |
| **XGBoost** | Meilleur algorithme de prédiction |
| **Snowflake ML (ExperimentTracking)** | Suivi des expériences & hyperparamètres |
| **Snowflake Model Registry** | Stockage & versioning du modèle |
| **Streamlit in Snowflake** | Application interactive de prédiction |
| **Altair** | Visualisations EDA |
| **Amazon S3** | Source des données brutes (JSON) |

---

## Tables créées dans Snowflake

| Table | Description |
|-------|-------------|
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE` | Dataset original (1090 lignes, 545 uniques) |
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_RAW` | Données JSON brutes |
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_TEST_DATA` | Données de test (features + prix réel) |
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_PREDICTIONS` | Prédictions vs valeurs réelles |
| `HOUSE_PRICE_DB.ML_SCHEMA.SCALER_PARAMS` | Paramètres du StandardScaler (mean, std) |

**Model Registry :** `HOUSE_PRICE_PREDICTOR` (versioning automatique)  
**Experiment :** `House_Price_Experiment` (48 runs trackées)

---

## Auteurs
Fall Magueye - Seynabou Sene - Mame Diarra Ndiaye

*MBAESG — Promotion MBIAD — Classe 02*

---

*Livrable : **MBAESG_MBDIA_O2_EVALUATION_DATAENGINEER_MLOPS***
