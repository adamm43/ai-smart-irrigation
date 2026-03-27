# 💧 Smart Irrigation AI
### Optimisation de l'irrigation intelligente par Machine Learning

---

## 📋 Description du projet

Ce projet académique vise à prédire le **volume d'eau nécessaire pour l'irrigation** (en litres/jour) en fonction des conditions climatiques, du type de culture et des caractéristiques du sol.

L'objectif est de réduire le gaspillage d'eau en agriculture grâce à des modèles de Machine Learning et de Deep Learning capables d'estimer avec précision les besoins hydriques d'un champ.

---

## 🎯 Problématique

> *"Comment optimiser la quantité d'eau utilisée pour l'irrigation afin de maximiser le rendement agricole tout en minimisant le gaspillage hydrique ?"*

L'agriculture représente environ **70% de la consommation mondiale d'eau douce**. Une irrigation mal calibrée entraîne :
- Un gaspillage d'eau important
- Une dégradation des sols (sur-irrigation)
- Une baisse des rendements (sous-irrigation)

Ce projet propose un modèle prédictif capable de répondre à la question : **"Combien de litres d'eau faut-il apporter aujourd'hui ?"**

---

## 📁 Structure du projet

```
smart_irrigation/
│
├── data/
│   ├── raw/
│   │   └── smart_irrigation_4000.csv    ← Dataset principal (4000 lignes)
│   └── processed/                        ← Données traitées (auto-généré)
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py              ← Chargement, nettoyage, preprocessing
│   ├── regression_model.py              ← LR, Ridge, RF, GBM + tuning + CV
│   ├── neural_network.py                ← Réseau de neurones (Keras/TensorFlow)
│   ├── evaluation.py                    ← Métriques, leaderboard, résidus
│   ├── visualizations.py                ← Toutes les fonctions de visualisation
│   └── utils.py                         ← Utilitaires (logging, sauvegarde)
│
├── config/
│   ├── __init__.py
│   └── config.py                        ← Configuration centrale (paths, hyperparamètres)
│
├── models/                              ← Modèles sauvegardés (.pkl, .keras)
│
├── app_dashboard.py                     ← Dashboard Streamlit interactif
├── main.py                              ← Pipeline complet en ligne de commande
├── requirements.txt                     ← Dépendances Python
└── README.md                            ← Ce fichier
```

---

## 📊 Dataset

**Fichier :** `smart_irrigation_4000.csv`  
**Taille :** 4000 lignes × 14 colonnes  
**Source :** Dataset généré avec des paramètres agronomiques réels

### Features (variables d'entrée)

| Feature | Type | Description |
|---------|------|-------------|
| `temperature_C` | Numérique | Température ambiante (°C) |
| `humidity_%` | Numérique | Humidité relative de l'air (%) |
| `soil_moisture_%` | Numérique | Humidité du sol (%) |
| `rainfall_mm` | Numérique | Précipitations (mm) |
| `solar_radiation_Wm2` | Numérique | Rayonnement solaire (W/m²) |
| `wind_speed_kmh` | Numérique | Vitesse du vent (km/h) |
| `soil_pH` | Numérique | pH du sol |
| `field_area_ha` | Numérique | Surface du champ (hectares) |
| `days_since_rain` | Numérique | Jours depuis la dernière pluie |
| `soil_type` | Catégoriel | Type de sol (Sandy/Clay/Loam/Silt) |
| `crop_type` | Catégoriel | Type de culture (Wheat/Maize/Rice/Cotton/Soybean) |
| `growth_stage` | Catégoriel | Stade de croissance (Germination/Vegetative/Flowering/Maturity) |
| `season` | Catégoriel | Saison (Spring/Summer/Autumn/Winter) |

### Variable cible

| Feature | Type | Description |
|---------|------|-------------|
| `water_amount_liters` | Numérique | **Volume d'eau nécessaire (litres/jour)** |

---

## 🤖 Modèles implémentés

| Modèle | Type | Justification |
|--------|------|---------------|
| **Linear Regression** | Linéaire | Baseline — interprétable, rapide |
| **Ridge Regression** | Linéaire + L2 | Gère la multicolinéarité |
| **Random Forest** | Ensemble | Robuste, importance des features |
| **Gradient Boosting** | Ensemble | Puissant pour relations non-linéaires |
| **Neural Network** | Deep Learning | Capture les interactions complexes |

---

## 📈 Métriques d'évaluation

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **MSE** | Mean Squared Error | Plus bas = meilleur |
| **RMSE** | Root Mean Squared Error | En litres — plus bas = meilleur |
| **MAE** | Mean Absolute Error | Erreur moyenne en litres |
| **R²** | Coefficient de détermination | Plus proche de 1 = meilleur |
| **MAPE%** | Mean Absolute Percentage Error | Erreur en % |

---

## 🖥️ Dashboard Streamlit

Le dashboard interactif contient **6 onglets** :

| Onglet | Contenu |
|--------|---------|
| 📊 **Dataset** | Vue d'ensemble, statistiques, patterns d'irrigation |
| 🔍 **Analyse EDA** | Distributions, corrélations, boxplots, scatter plots |
| 🏆 **Comparaison Modèles** | Leaderboard, métriques visuelles, résidus, CV |
| 🧠 **Réseau de Neurones** | Courbes d'entraînement, architecture, historique |
| 🎯 **Prédiction Live** | Prédiction interactive avec interprétation (🟢🟡🔴) |
| 📈 **Feature Importance** | Importances RF/GBM, coefficients régression linéaire |

---

## 🚀 Installation et exécution

### 1. Prérequis

```bash
Python 3.9+
```

### 2. Installation des dépendances

```bash
pip3 install -r requirements.txt
```

### 3. Lancer le dashboard

```bash
cd smart_irrigation
python3 -m streamlit run app_dashboard.py
```

Ouvrir dans Chrome : `http://localhost:8501`

### 4. Pipeline complet (ligne de commande)

```bash
python3 main.py
```

---

## 🏗️ Architecture technique

### Préparation des données (`src/data_preparation.py`)
- Chargement et validation du dataset
- Suppression des doublons et valeurs manquantes
- **sklearn ColumnTransformer** :
  - Features numériques → `SimpleImputer(median)` + `StandardScaler`
  - Features catégorielles → `SimpleImputer(most_frequent)` + `OneHotEncoder`
- **Split train/test avant le fit** du préprocesseur (pas de data leakage)

### Modèles ML (`src/regression_model.py`)
- `RandomizedSearchCV` pour le tuning hyperparamètres (RF et GBM)
- `cross_val_score` pour la validation croisée k-fold
- Permutation importance pour les modèles linéaires

### Réseau de neurones (`src/neural_network.py`)
```
Input(26)
  → Dense(128) + BatchNormalization + ReLU + Dropout(0.3)
  → Dense(64)  + BatchNormalization + ReLU + Dropout(0.3)
  → Dense(32)  + BatchNormalization + ReLU + Dropout(0.15)
  → Dense(1, linear)

Optimizer : Adam (lr=0.001)
Loss      : Huber (robuste aux outliers)
Callbacks : EarlyStopping(patience=20) + ReduceLROnPlateau
```

---

## 📌 Règles d'irrigation (domaine métier)

Le modèle capture les règles agronomiques suivantes :

- 🌡️ **Température élevée** → plus d'eau (évapotranspiration)
- 💧 **Sol humide** → moins d'eau nécessaire
- 🌧️ **Pluie récente** → moins d'irrigation
- ☀️ **Fort rayonnement** → plus d'évaporation → plus d'eau
- 🌾 **Rice > Maïs > Blé** en besoin hydrique
- 🌸 **Floraison** = stade le plus sensible → besoin maximal
- ☀️ **Été > Printemps > Automne > Hiver** en besoin d'irrigation
- 🏜️ **Sol sableux** → retient moins l'eau → arrosage plus fréquent

---

## 👨‍💻 Technologies utilisées

- **Python 3.9**
- **pandas, numpy, scipy** — manipulation des données
- **scikit-learn** — modèles ML, pipeline, métriques
- **TensorFlow / Keras** — réseau de neurones
- **matplotlib, seaborn** — visualisations
- **Streamlit** — dashboard interactif
- **joblib** — sauvegarde des modèles

---

## 📝 Auteur

Projet académique — *Applications de l'Intelligence Artificielle*  
Sujet : **Optimisation de l'irrigation intelligente**

---
