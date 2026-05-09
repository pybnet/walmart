# Walmart — Weekly Sales Prediction

Predicting weekly sales across Walmart stores from economic and calendar features, using linear models with regularization and a tree-based benchmark.

## Project goals

- Build a regression model to estimate `Weekly_Sales` per store and per week
- Identify which features drive sales (coefficient interpretation)
- Mitigate overfitting on a small dataset using Ridge and Lasso regularization
- Benchmark linear models against tree-based ensembles (RandomForest, ExtraTrees, XGBoost)

## Dataset

- File: `data/Walmart_Store_sales.csv`
- Target: `Weekly_Sales`
- Features: `Store`, `Date`, `Holiday_Flag`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
- Size after cleaning: ~131 rows (small dataset → cross-validation is critical)

## Pipeline

1. **EDA & cleaning**
   - Drop rows where `Weekly_Sales` is missing (no imputation on the target)
   - Fill missing values: forward-fill for `Date`, median for numerical columns, mode for `Store`
   - Outlier removal on `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` using the `[mean ± 3σ]` rule
   - Feature engineering on `Date` → `Year`, `Month`, `Day`, `Week_Number`
   - Correlation analysis → drop `Year` (collinear with `Fuel_Price`) and `Week_Number` (collinear with `Month`)

2. **Preprocessing (sklearn `ColumnTransformer`)**
   - Categorical: `Store`, `Holiday_Flag`, `Month` → `OneHotEncoder(drop="first")`
   - Numerical: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment` → `StandardScaler`
   - Train/test split: 80/20, `random_state=42`

3. **Models**
   - Baseline: `LinearRegression`
   - Regularized: `Ridge` and `Lasso` tuned via `GridSearchCV` (5-fold CV, scoring=`r2`)
   - Ensembles: `RandomForestRegressor`, `ExtraTreesRegressor`, `XGBRegressor`

4. **Evaluation**
   - Metrics: R², MAE, RMSE
   - 10-fold cross-validation on the best regularized estimators

## Results

| Model                | R² Train | R² Test | CV R²  | CV Std |
|----------------------|---------:|--------:|-------:|-------:|
| Linear Regression    | 0.9824   | 0.8950  | —      | —      |
| Ridge (α=0)          | 0.9824   | 0.8949  | 0.9379 | 0.0389 |
| **Lasso (α=780.1)**  | 0.9809   | **0.9035** | **0.9460** | 0.0403 |
| RandomForest         | 0.8664   | 0.5200  | 0.4454 | 0.1066 |
| ExtraTrees           | 0.8661   | 0.7886  | 0.6367 | 0.0544 |
| XGBoost              | 0.9800   | 0.7519  | 0.7030 | 0.0980 |

**Best model: Lasso regression (α = 780.1)** — R² test = 0.9035, CV R² = 0.946.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook project_walmart_sales.ipynb
```

Place `Walmart_Store_sales.csv` in a `data/` folder next to the notebook.
