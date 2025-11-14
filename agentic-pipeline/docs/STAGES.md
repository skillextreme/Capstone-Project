# Stage-by-Stage Guide

This document provides a detailed walkthrough of each pipeline stage with examples.

---

## Stage 1: Summarizer

### Purpose
Generate factual summaries of raw data files without any analysis or interpretation.

### What It Does

For each CSV/JSON file:

1. **Schema Analysis**
   - Column names
   - Inferred types (numeric, categorical, date, string, boolean)
   - Example values

2. **Statistical Summaries**
   - **Numeric columns:** mean, std, min, max, median, quartiles
   - **Categorical columns:** cardinality, top 10 values, mode
   - **Date columns:** min/max dates, date range
   - **All columns:** null count, null rate

3. **Key Detection**
   - **Primary key candidates:** Columns with >95% unique values
   - **Foreign key candidates:** Common join columns (state, year, crop, etc.)
   - **Composite keys:** Common combinations (state+year, state+year+crop)

### Example Input

`data/raw/crop_yield.csv`:
```csv
state,year,crop,area_hectares,production_tonnes,yield_kg_per_hectare
Punjab,2020,Rice,3000000,10500000,3500
Punjab,2021,Rice,3100000,11000000,3548
Haryana,2020,Wheat,2500000,9000000,3600
...
```

### Example Output

`data/summaries/crop_yield.summary.json`:
```json
{
  "file_name": "crop_yield.csv",
  "row_count": 15000,
  "column_count": 6,
  "columns": [
    {
      "name": "state",
      "type": "categorical",
      "null_rate": 0.01,
      "cardinality": 35,
      "top_values": {
        "Punjab": 450,
        "Haryana": 420,
        "UP": 580
      },
      "is_key_candidate": false,
      "uniqueness_ratio": 0.002
    },
    {
      "name": "year",
      "type": "numeric",
      "null_rate": 0.0,
      "mean": 2015.5,
      "std": 5.8,
      "min": 2000,
      "max": 2023,
      "median": 2015,
      "is_key_candidate": false
    },
    {
      "name": "yield_kg_per_hectare",
      "type": "numeric",
      "null_rate": 0.05,
      "mean": 3200.5,
      "std": 850.2,
      "min": 800,
      "max": 6500,
      "is_key_candidate": false
    }
  ],
  "candidate_keys": {
    "primary": [
      ["state", "year", "crop"]
    ],
    "foreign": ["state", "year", "crop"]
  }
}
```

### Running Stage 1

```bash
# Process all files in data/raw/
python src/main.py --mode stage1

# Or use the module directly
python -m src.stage1.summarizer --data-dir data/raw --output-dir data/summaries

# With sampling for large files
python -m src.stage1.summarizer --sample-size 10000
```

### Customization

Edit `config/pipeline_config.yaml`:

```yaml
summarizer:
  sample_size: 1000  # null = all rows
  numeric_threshold: 0.95  # 95% must parse as number
  max_categorical_cardinality: 100  # max unique values
  top_k_values: 10  # how many top values to include
  key_uniqueness_threshold: 0.95  # 95% unique for primary key
```

---

## Stage 2: Task Suggester

### Purpose
Propose feasible analysis tasks based on available data.

### What It Does

Analyzes all summaries and suggests:

1. **Prediction Tasks**
   - Target: Numeric column (yield, price, production)
   - Features: Joinable files with candidate features
   - Time structure: Year column for time-series modeling
   - Example: "Predict crop yield using rainfall and fertilizer data"

2. **Descriptive Tasks**
   - Aggregations: Mean yield by state, total production by year
   - Rankings: Top 10 states by production
   - Example: "Find largest producing state per crop per year"

3. **Clustering Tasks**
   - Entities: States, crops, districts
   - Features: Multiple numeric columns
   - Example: "Cluster states by multi-year yield profiles"

### How It Works

**Rule-Based Heuristics:**

1. **For Prediction:**
   - Find files with numeric columns matching keywords (yield, production, price)
   - Check for time structure (year column exists)
   - Find other files with common keys (for joining)
   - Calculate key overlap (Jaccard similarity)
   - Score feasibility: high if has time + joinable features

2. **For Descriptive:**
   - Find numeric + categorical columns
   - Check cardinality of categorical (must be manageable)
   - Suggest groupby/aggregation combinations

3. **For Clustering:**
   - Find files with ≥3 numeric features
   - Find entity columns (state, crop, district)
   - Suggest clustering by entity

### Example Output

`data/tasks.json`:
```json
[
  {
    "task_id": "T1",
    "type": "prediction",
    "subtype": "regression",
    "description": "Predict yield_kg_per_hectare using historical data",
    "target_variable": "yield_kg_per_hectare",
    "target_file": "crop_yield.csv",
    "required_files": [
      "crop_yield.csv",
      "rainfall.csv",
      "fertilizer_usage.csv"
    ],
    "required_keys": ["state", "year", "crop"],
    "time_series": true,
    "features": [
      "rainfall.csv:rainfall_mm",
      "fertilizer_usage.csv:npk_kg_per_hectare"
    ],
    "feasibility": "high"
  },
  {
    "task_id": "T2",
    "type": "descriptive",
    "subtype": "aggregation",
    "description": "Analyze production_tonnes by state",
    "target_variable": "production_tonnes",
    "group_by": ["state"],
    "aggregation": "mean",
    "required_files": ["crop_yield.csv"],
    "feasibility": "high"
  }
]
```

### Running Stage 2

```bash
# Generate task suggestions
python src/main.py --mode stage2

# Or directly
python -m src.stage2.task_suggester --summaries-dir data/summaries --output data/tasks.json

# Limit number of suggestions
python -m src.stage2.task_suggester --max-suggestions 10
```

### Customization

```yaml
task_suggester:
  min_files_for_join: 2  # min files needed for join tasks
  min_key_overlap: 0.5  # 50% key overlap for joinability
  max_suggestions: 5  # max tasks to propose
```

---

## Stage 3: Planner

### Purpose
Create a reproducible data plan: join files, normalize keys, engineer features.

### What It Does

1. **Load & Normalize**
   ```python
   # Load all required files
   yields = load_csv("crop_yield.csv")
   rainfall = load_csv("rainfall.csv")

   # Normalize keys
   yields['state'] = normalize(yields['state'])  # "UP" → "Uttar Pradesh"
   rainfall['state'] = normalize(rainfall['state'])
   ```

2. **Merge Files**
   ```python
   # Join on common keys
   merged = yields.merge(rainfall, on=['state', 'year', 'crop'], how='left')

   # Log join cardinality
   logger.info(f"Rows: {len(yields)} → {len(merged)}")
   ```

3. **Feature Engineering**
   ```python
   # Lag features (previous year's yield)
   merged['yield_lag1'] = merged.groupby(['state', 'crop'])['yield'].shift(1)
   merged['yield_lag2'] = merged.groupby(['state', 'crop'])['yield'].shift(2)

   # Rolling statistics (3-year average)
   merged['yield_roll3_mean'] = merged.groupby(['state', 'crop'])['yield'].rolling(3).mean()

   # Rainfall anomaly
   merged['rainfall_anomaly'] = merged['rainfall_mm'] - merged.groupby('state')['rainfall_mm'].transform('mean')
   ```

4. **Data Cleaning**
   ```python
   # Drop rows with >50% missing
   missing_ratio = merged.isna().sum(axis=1) / len(merged.columns)
   merged = merged[missing_ratio <= 0.5]

   # Impute remaining missing values
   merged.fillna(merged.median(), inplace=True)
   ```

5. **Save Outputs**
   ```python
   # Save merged data
   merged.to_parquet("data/intermediate/T1_merged.parquet")

   # Save join plan
   plan = {
       "task_id": "T1",
       "files_used": ["crop_yield.csv", "rainfall.csv"],
       "join_keys": ["state", "year", "crop"],
       "rows": len(merged),
       "columns": list(merged.columns)
   }
   save_json(plan, "data/intermediate/T1_plan.json")
   ```

### Example: Join Graph

```
crop_yield.csv (10,000 rows)
    │
    ├─ join on [state, year, crop] ─→ rainfall.csv (8,000 rows)
    │                                       │
    │                                       ↓
    │                                  merged_1 (10,000 rows)
    │                                       │
    └─ join on [state, year] ───────────→  fertilizer.csv (12,000 rows)
                                            │
                                            ↓
                                       final_merged (10,000 rows)
```

### Running Stage 3

```bash
# For a specific task
python src/main.py --mode stage3 --task-id T1

# Or directly
python -m src.stage3.planner --task-file data/tasks.json --task-id T1
```

### Customization

```yaml
planner:
  join_type: "left"  # inner, left, right, outer
  drop_na_threshold: 0.5  # drop rows with >50% missing

  # Key normalization rules
  key_normalizations:
    state:
      "UP": "Uttar Pradesh"
      "TN": "Tamil Nadu"

  # Feature engineering
  features:
    create_lags: true
    lag_periods: [1, 2]  # t-1, t-2
    create_rolling: true
    rolling_windows: [3, 5]  # 3-year, 5-year windows
```

---

## Stage 4: Executor

### Purpose
Run the analysis, train models, generate outputs.

### What It Does

### For Prediction Tasks

1. **Prepare Data**
   ```python
   # Load merged data
   df = load_parquet("data/intermediate/T1_merged.parquet")

   # Extract features (numeric columns)
   X = df.select_dtypes(include=[np.number])
   X = X.drop(columns=['yield_kg_per_hectare', 'state', 'year'])  # drop target and IDs

   # Extract target
   y = df['yield_kg_per_hectare']

   # Impute missing
   X.fillna(X.median(), inplace=True)
   ```

2. **Split Data**
   ```python
   # Time-based split (train on ≤2020, test on >2020)
   train_mask = df['year'] <= 2020
   test_mask = df['year'] > 2020

   X_train, y_train = X[train_mask], y[train_mask]
   X_test, y_test = X[test_mask], y[test_mask]
   ```

3. **Train Models**
   ```python
   # Linear Regression (Ridge)
   ridge = Ridge(alpha=1.0)
   ridge.fit(X_train, y_train)

   # XGBoost
   xgb_model = XGBRegressor(n_estimators=100, max_depth=6)
   xgb_model.fit(X_train, y_train)
   ```

4. **Evaluate**
   ```python
   # Predictions
   y_pred_test = xgb_model.predict(X_test)

   # Metrics
   mae = mean_absolute_error(y_test, y_pred_test)
   rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
   r2 = r2_score(y_test, y_pred_test)
   mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

   print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")
   ```

5. **Visualize**
   - Actual vs Predicted scatter plot
   - Residuals by state (bar chart)
   - Time series (actual vs predicted over years)
   - Feature importance (XGBoost importances)
   - Error distribution (histogram + Q-Q plot)

6. **Save Outputs**
   - `T1_metrics.json` - All metrics
   - `T1_predictions.csv` - Predictions with residuals
   - `T1_model_card.json` - Model documentation
   - `plots/` - All visualizations

### Example Metrics Output

`data/outputs/T1_metrics.json`:
```json
{
  "task_id": "T1",
  "timestamp": "2024-01-15T10:30:00",
  "best_model": "gradient_boosting",
  "models": {
    "linear_regression": {
      "train": {"mae": 250.5, "rmse": 320.1, "r2": 0.72},
      "test": {"mae": 280.2, "rmse": 350.8, "r2": 0.68}
    },
    "gradient_boosting": {
      "train": {"mae": 180.3, "rmse": 230.5, "r2": 0.85},
      "test": {"mae": 220.1, "rmse": 280.3, "r2": 0.78}
    }
  }
}
```

### Running Stage 4

```bash
# For a specific task
python src/main.py --mode stage4 --task-id T1

# Or directly
python -m src.stage4.executor --task-file data/tasks.json --task-id T1 --data-path data/intermediate/T1_merged.parquet
```

### Customization

```yaml
executor:
  split:
    method: "time"  # or "random"
    test_split_year: 2020
    test_size: 0.2  # for random split

  models:
    linear_regression:
      enabled: true
      regularization: "ridge"
      alpha: 1.0

    gradient_boosting:
      enabled: true
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1

  visualization:
    plots:
      actual_vs_predicted: true
      residuals_by_segment: true
      time_series: true
      feature_importance: true
```

---

## Complete Example Workflow

```bash
# 1. Place data files
cp ~/Downloads/crop_yield.csv data/raw/
cp ~/Downloads/rainfall.csv data/raw/

# 2. Run pipeline
python src/main.py --mode full

# Output:
# ✓ Stage 1: Summarized 2 files
# ✓ V1: All summaries valid
# ✓ Stage 2: Generated 3 task suggestions
# ✓ V2: Selected 1 task (T1)
# ✓ Stage 3: Merged data (8,500 rows, 25 columns)
# ✓ V3: Join valid, no leakage detected
# ✓ Stage 4: Trained 2 models, best = gradient_boosting (MAE: 220.1, R²: 0.78)
# ✓ V4: Metrics valid, model card complete

# 3. View outputs
ls data/outputs/
# T1_metrics.json
# T1_predictions.csv
# T1_model_card.json
# plots/actual_vs_pred_gradient_boosting.png
# plots/residuals_by_state_gradient_boosting.png
# ...
```

---

## Tips & Best Practices

1. **Start small:** Test with sample data first
2. **Inspect summaries:** Review Stage 1 outputs before proceeding
3. **Iterate on tasks:** Modify `data/tasks.json` manually if needed
4. **Check verifications:** Review verification reports in each stage
5. **Tune models:** Edit `config/pipeline_config.yaml` for hyperparameters
6. **Visualize early:** Generate plots to understand data before modeling
