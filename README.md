# 🏠 House Price Prediction — Pipeline ML complet sur Snowflake

> **Évaluation Data Engineering & Machine Learning avec Snowflake**  
> **MBAESG_MBIAD_O2_EVALUATION_DATAENGINEER_MLOPS**

---

## 📋 Description du projet

Ce projet implémente un pipeline complet de **Data Engineering et Machine Learning** directement dans l'environnement **Snowflake**, sans extraction de données vers des systèmes externes. Il utilise les données de caractéristiques de maisons pour prédire leur prix de vente.

**Dataset source :** `s3://logbrain-datalake/datasets/house_price/`  
**Format :** JSON  
**Nombre de lignes :** 1 090

---

## 🗂️ Contenu du dépôt

```
.
├── HOUSE_PRICE_PREDICTION_NOTEBOOK.ipynb   # Notebook Snowflake — pipeline ML complet
├── streamlit_app.py                        # Application Streamlit de prédiction
└── README.md                               # Ce fichier — documentation & analyse
```

---

## 📦 Dataset

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

## 🔧 Architecture du pipeline

```
S3 (logbrain-datalake) — Format JSON
          │
          ▼
    External Stage
          │
          ▼
    Table HOUSE_PRICE (1090 lignes)
          │
          ▼
    EDA (Snowpark + Altair + Streamlit)
          │
          ▼
    Feature Engineering
    ├── Encodage binaire (yes/no → 0/1)
    ├── Encodage ordinal (FURNISHINGSTATUS)
    ├── Normalisation (StandardScaler → table SCALER_PARAMS)
    └── Split Train 80% / Test 20%
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

## 📊 Analyse des performances des modèles

### Étape 1 — Comparaison initiale des modèles (sans optimisation)

| Modèle | Train MAE | Test MAE | Train RMSE | Test RMSE | Train R² | Test R² |
|--------|-----------|----------|------------|-----------|----------|---------|
| Régression Linéaire | 39 696 | 40 253 | 53 640 | 53 985 | 0.6740 | 0.6732 |
| Random Forest Regressor | 7 263 | 19 587 | 13 651 | 32 513 | 0.9789 | 0.8815 |
| **XGBoost Regressor** | **2 630** | **12 757** | **5 036** | **27 517** | **0.9971** | **0.9151** |

**Observations :**
- La **Régression Linéaire** obtient un R² de 0.67, insuffisant — les relations entre features et prix ne sont pas purement linéaires.
- **Random Forest** améliore significativement les performances (R² = 0.88) mais présente un sur-apprentissage marqué (écart train/test important).
- **XGBoost** est le meilleur modèle initial avec un R² de 0.9151 sur le test et le MAE le plus faible (12 757 USD).

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

**Algorithme sélectionné : XGBoost Regressor**

| Hyperparamètre | Valeur optimale |
|----------------|----------------|
| `n_estimators` | 100 |
| `max_depth` | 7 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `random_state` | 42 |

**Performances finales :**

| Métrique | Valeur |
|----------|--------|
| **R² (test)** | **0.9172** |
| **MAE (test)** | **14 687 USD** |
| **RMSE (test)** | **27 173 USD** |

---

### Interprétation des résultats

- Un **R² de 0.9172** signifie que le modèle explique **91.7% de la variance** des prix — excellente performance.
- Un **MAE de 14 687 USD** signifie qu'en moyenne, le modèle se trompe de ~14 700 USD sur le prix prédit.
- Le **RMSE de 27 173 USD** indique que les erreurs importantes restent contenues.

### Features les plus influentes (d'après XGBoost)

1. **AREA** — La surface est le facteur le plus déterminant du prix
2. **BATHROOMS** — Lié à la qualité et la taille du bien
3. **AIRCONDITIONING** — Indicateur de confort premium
4. **STORIES** — Nombre d'étages
5. **PREFAREA** — Zone géographique privilégiée
6. **FURNISHINGSTATUS** — Meublé > semi-meublé > non meublé
7. **PARKING** — Nombre de places de stationnement

---

## 🚀 Instructions d'exécution

### Prérequis Snowflake
- Rôle avec droits `CREATE DATABASE`, `CREATE SCHEMA`, `CREATE STAGE`, `CREATE TABLE`, `CREATE MODEL`
- Accès au bucket S3 `s3://logbrain-datalake/`

### Packages Python à installer (menu Packages du notebook)
```
xgboost
snowflake-ml-python
```
> `scikit-learn`, `pandas`, `altair` et `streamlit` sont inclus par défaut.

### Ordre d'exécution
1. Ouvrir `HOUSE_PRICE_PREDICTION_NOTEBOOK.ipynb` dans **Snowflake Notebooks**
2. Installer les packages requis
3. Exécuter les cellules dans l'ordre (étapes 1 à 10)
4. Créer une **Streamlit App** dans Snowflake et coller le contenu de `streamlit_app.py`

---

## 🏗️ Technologies utilisées

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

## 📁 Tables créées dans Snowflake

| Table | Description |
|-------|-------------|
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE` | Dataset original (1090 lignes) |
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_RAW` | Données JSON brutes |
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_TEST_DATA` | Données de test (features + prix réel) |
| `HOUSE_PRICE_DB.ML_SCHEMA.HOUSE_PRICE_PREDICTIONS` | Prédictions vs valeurs réelles |
| `HOUSE_PRICE_DB.ML_SCHEMA.SCALER_PARAMS` | Paramètres du StandardScaler (mean, std) |

**Model Registry :** `HOUSE_PRICE_PREDICTOR` (versioning automatique)  
**Experiment :** `House_Price_Experiment` (48 runs trackées)

---

## 👥 Auteurs
Fall Magueye - Seynabou Sene - Mame Diarra Ndiaye

*MBAESG — Promotion MBIAD — Classe O2*

---

*Livrable : **MBAESG_MBIAD_O2_EVALUATION_DATAENGINEER_MLOPS***
