# SEATLE — Predicting Building Energy & GHG Emissions

Supervised machine learning project for predicting energy consumption and greenhouse gas emissions of Seattle buildings (2016 benchmark data). Two regression models target total energy use and total CO2 emissions respectively.

---

## Objective

Build and optimize predictive models for estimating two critical environmental metrics of non-residential buildings in Seattle: **total energy consumption** (SiteEnergyUse in kBtu) and **total GHG emissions** (metric tons CO2e). This analysis supports building energy efficiency initiatives and carbon footprint assessments.

**Key deliverables:**
- Exploratory data analysis with outlier detection and feature engineering
- Supervised regression models optimized via GridSearchCV and 5-fold cross-validation
- Feature importance analysis to identify key drivers of energy and emissions

---

## Features

✅ **Data Cleaning & Preprocessing** — Remove outliers (95th percentile), handle missing values, standardize numeric features with RobustScaler  
✅ **Feature Engineering** — Build 12+ derived features (building age, area ratios, energy source flags, activity count)  
✅ **Two Target Models** — Separate pipelines for energy (SiteEnergyUse) and emissions (TotalGHGEmissions)  
✅ **Model Optimization** — GridSearchCV on GradientBoosting (energy) and RandomForest (emissions) with 5-fold CV  
✅ **Performance Evaluation** — R², MAE, MSE metrics across train/test splits  
✅ **Interpretability** — Feature importance plots and residual analysis by building size  

---

## Architecture

```
Data Ingestion (CSV)
       ↓
   Exploratory Data Analysis
       ↓
   Data Cleaning (outliers, missing)
       ↓
   Feature Engineering (12+ features)
       ↓
   Train/Test Split (80/20, stratified)
       ├─→ Energy Model (GradientBoosting)
       └─→ Emissions Model (RandomForest)
       ↓
   Hyperparameter Tuning (GridSearchCV)
       ↓
   Model Evaluation & Feature Importance
```

---

## Project Structure

```
SEATLE/
├── 📄 README.md                          # This file
├── 📄 requirement.txt                    # Python dependencies
│
├── 📂 notebook/
│   └── 📓 P3_template_modelistation_supervisee.ipynb    # Full pipeline (EDA → modeling)
│
├── 📂 data/
│   ├── 📊 2016_Building_Energy_Benchmarking.csv         # Raw dataset (3376 buildings, 46 cols)
│   ├── 📊 df_total_ENERGY.csv                           # Energy predictions (test set)
│   └── 📊 df_total_CO2.csv                              # Emissions predictions (test set)
│
├── 📂 images/
│   ├── 📈 importance energy.png              # Feature importance for energy model
│   ├── 📈 importance CO2.png                 # Feature importance for emissions model
│   ├── 📈 corrmat.png                        # Correlation heatmap (numeric features)
│   ├── 📈 Distribution TotalGHGEmissions_*.png
│   └── ... (12+ visualizations of distributions, residuals, predictions)
│
└── 📂 docs/
    └── 📑 P3 - SEATTLE OPENCLASSROOMS.pptx   # Final presentation slides
```

---

## Technology Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.x | Core language |
| pandas | 2.3.2 | Data manipulation, EDA |
| scikit-learn | 1.6.1 | ML models, preprocessing, GridSearchCV |
| matplotlib | 3.9.4 | Plotting |
| seaborn | 0.13.2 | Statistical visualization |
| numpy | 2.0.2 | Numerical computing |
| Jupyter | 6.30.1 | Interactive notebook environment |

---

## Installation & Usage

### Prerequisites

- **Python 3.8+** (tested with Python 3.x)
- **pip** or **conda** for package management
- **Jupyter Notebook** or **Jupyter Lab**
- 2+ GB free disk space (dataset + environment)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/RandomFab/SEATLE.git
   cd SEATLE
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv env
   source env/bin/activate          # macOS/Linux
   # or
   env\Scripts\activate              # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```
   
   Or install key packages manually:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy jupyter scipy
   ```

4. **Launch Jupyter**
   ```bash
   jupyter notebook
   ```
   
   Then open `notebook/P3_template_modelistation_supervisee.ipynb`

5. **Run cells sequentially** from top to bottom:
   - **Cells 1–71:** EDA, data cleaning, outlier removal (95th percentile)
   - **Cells 72–112:** Feature engineering (12+ new features)
   - **Cells 113–162:** Energy model training, evaluation, GridSearchCV
   - **Cells 163–173:** Emissions model training, evaluation, GridSearchCV

---

## Model Results

### Energy Consumption (SiteEnergyUse in kBtu)

**Best Model:** Gradient Boosting Regressor (optimized via GridSearchCV)

| Metric | Train CV | Test Set |
|--------|----------|----------|
| **R² Score** | 0.745 | **0.73** |
| **MAE (kBtu)** | ~2.2M | **~1.9M** |
| **MSE** | — | 1.46e13 |

**Best Hyperparameters (GridSearchCV):**
- `n_estimators`: 250
- `learning_rate`: 0.05
- `max_depth`: 5
- `subsample`: 0.6
- `criterion`: 'squared_error'

**Top 5 Feature Drivers:**
1. LargestPropertyUseTypeGFA (building footprint of primary use)
2. sf_floor (avg square footage per floor)
3. NumberofFloors
4. SecondLargestPropertyUseTypeGFA (secondary use footprint)
5. ENERGYSTARScore

---

### GHG Emissions (TotalGHGEmissions in metric tons CO2e)

**Best Model:** Random Forest Regressor (optimized via GridSearchCV)

| Metric | Train CV | Test Set |
|--------|----------|----------|
| **R² Score** | 0.519 | **0.58** |
| **MAE (tons CO2)** | ~66–70 | **~62** |
| **MSE** | — | 14,445 |

**Best Hyperparameters (GridSearchCV):**
- `n_estimators`: 1000
- `max_depth`: 100
- `max_features`: 'sqrt'
- `min_samples_leaf`: 1
- `bootstrap`: False

**Top 5 Feature Drivers:**
1. LargestPropertyUseTypeGFA
2. sf_floor
3. NumberofFloors
4. SecondLargestPropertyUseTypeGFA
5. ENERGYSTARScore

---

## Key Insights

**Data Processing:**
- Started with 3,376 buildings; retained 1,476 after removing non-residential, outliers (95th percentile), and non-compliant records
- Outlier threshold for energy: 63.3M kBtu; for emissions: 1,522 tons CO2e
- ENERGYSTARScore: imputed ~840 missing values using uniform random sampling within ±1σ

**Model Performance:**
- Energy model (R² 0.73) performs significantly better than emissions model (R² 0.58)
  - Reason: Energy metrics are more directly correlated with physical building attributes
  - Emissions depend on regional grid mix and energy sources, introducing noise
- Both models show no systematic bias across different building size ranges (residual analysis by deciles)
- GradientBoosting outperforms Ridge and RandomForest for energy; RandomForest edges out GradientBoosting for emissions

**Feature Importance:**
- Building geometry (floor count, footprint) dominates both models
- Energy source availability (HasGas, HasSteam) and building age have secondary effects
- One-hot encoded property type (Hotel, Office, Hospital, etc.) provides modest improvements

---

## How to Reproduce Results

1. Run all cells in the Jupyter notebook sequentially
2. Cells automatically:
   - Load raw CSV (2016_Building_Energy_Benchmarking.csv)
   - Clean and filter data
   - Engineer features
   - Train/test split (80/20)
   - Execute GridSearchCV for hyperparameter tuning
   - Generate predictions and save to `df_total_ENERGY.csv` and `df_total_CO2.csv`
3. Charts, importance plots, and prediction scatter plots are rendered inline

**Note:** GridSearchCV operations take ~5–10 minutes depending on system. Use `n_jobs=-1` to parallelize across all CPU cores.

---

## Limitations & Future Work

**Current Limitations:**
- R² score for emissions (0.58) indicates room for improvement; likely due to complex, non-linear dependencies on grid composition and energy sources
- Dataset limited to 2016 Seattle; not generalizable to other cities or years
- Cross-validation uses only 5 folds; increasing to 10 may yield more stable estimates
- ENERGYSTARScore imputation is simplistic (uniform random); alternative: k-NN or iterative imputation

**Potential Improvements:**
- Include external features: weather data, energy grid composition, gas/electricity prices
- Ensemble multiple models (Stacking, Voting Regressor)
- Experiment with non-linear models (XGBoost, LightGBM)
- Perform stratified train/test split by building type or size quartile
- Address multicollinearity among GFA features using dimensionality reduction (PCA)

---

## Author

**RandomFab - Fabien BARDOUIL**

OpenClassrooms Project P3 — Supervised Machine Learning (Data Science Path)

---

## License

Open for educational and non-commercial use. Refer to OpenClassrooms project guidelines.

---

## Acknowledgments

- Dataset: 2016 Building Energy Benchmarking (Seattle public data)
- OpenClassrooms for project framework and evaluation rubric
