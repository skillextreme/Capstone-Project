# Agricultural Data Analysis: 4-Stage Agentic Pipeline

## Project Overview

This capstone project implements a **4-stage agentic pipeline** for automated agricultural data analysis. The system analyzes raw CSV/JSON datasets, proposes feasible analyses, joins and prepares data, and trains machine learning models - all with automated verification at each stage.

**Key Innovation:** Stateless, reproducible pipeline with verification gates that ensures data quality and model transparency throughout the analysis process.

## What This System Does

### Input
Raw agricultural datasets (CSV/JSON):
- Crop yields (state, year, crop, yield, area, production)
- Rainfall data (state, year, rainfall_mm)
- Fertilizer usage (state, year, npk_kg_per_hectare)
- Any tabular agricultural data with common keys

### Output
Complete analysis package:
- **Predictions:** CSV with actual vs predicted values
- **Metrics:** MAE, RMSE, R², MAPE for all models
- **Visualizations:** 5+ publication-ready plots
- **Model Cards:** Full documentation of features, config, performance
- **Verification Reports:** Quality checks at each stage

### Automation Level
- **Fully automated:** Place data in `data/raw/`, run one command
- **Interactive mode:** Review and select suggested tasks
- **Configurable:** Tune models, features, splits via YAML config

## Architecture Highlights

### 4 Stages + 4 Verifications

1. **Stage 1: Summarizer** → **V1: Schema Check**
   - Analyzes CSV/JSON files for schema, statistics, keys
   - Validates column types, ranges, null rates

2. **Stage 2: Task Suggester** → **V2: Human Adjudication**
   - Proposes prediction, descriptive, clustering tasks
   - User selects which task(s) to execute

3. **Stage 3: Planner** → **V3: Join & Leakage Check**
   - Joins files, normalizes keys, engineers features
   - Validates join cardinality, coverage, time-based splits

4. **Stage 4: Executor** → **V4: Metrics Check**
   - Trains models (Ridge, XGBoost), generates outputs
   - Validates metrics, residuals, model transparency

### Design Principles

1. **Stateless:** Each stage writes artifacts to disk (JSON, Parquet, CSV)
2. **Modular:** Stages run independently or as full pipeline
3. **Reproducible:** Same inputs → same outputs (random seeds controlled)
4. **Verifiable:** Automated checks at each stage (non-blocking)
5. **Transparent:** All decisions logged, model cards generated

## Technology Stack

| Component | Technologies |
|-----------|--------------|
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualization** | matplotlib, seaborn |
| **Configuration** | PyYAML, python-dotenv |
| **Storage** | Parquet (intermediate), JSON (metadata), CSV (outputs) |

## Project Structure

```
agentic-pipeline/
│
├── README.md                    # Main documentation (setup, usage)
├── QUICKSTART.md                # 5-minute getting started guide
├── PROJECT_SUMMARY.md           # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
│
├── config/
│   ├── pipeline_config.yaml     # All stage configurations
│   └── .env.example             # API keys template
│
├── data/                        # Data storage (gitignored)
│   ├── raw/                     # Original CSV/JSON files
│   ├── summaries/               # Stage 1 outputs (JSON)
│   ├── intermediate/            # Stage 3 outputs (Parquet, JSON)
│   └── outputs/                 # Stage 4 outputs (CSV, JSON, PNG)
│       └── plots/               # Visualizations
│
├── src/                         # Source code
│   ├── main.py                  # Pipeline orchestrator (CLI entry point)
│   ├── config.py                # Configuration management
│   │
│   ├── stage1/
│   │   ├── __init__.py
│   │   └── summarizer.py        # File analysis (schema, stats, keys)
│   │
│   ├── stage2/
│   │   ├── __init__.py
│   │   └── task_suggester.py    # Task proposal (prediction, descriptive, clustering)
│   │
│   ├── stage3/
│   │   ├── __init__.py
│   │   ├── planner.py           # Join orchestration, feature engineering
│   │   └── normalizer.py        # Key normalization (UP → Uttar Pradesh)
│   │
│   ├── stage4/
│   │   ├── __init__.py
│   │   ├── executor.py          # Main execution logic
│   │   ├── models.py            # ML models (Ridge, XGBoost, RF)
│   │   └── visualizer.py        # All plots
│   │
│   ├── verifiers/
│   │   ├── __init__.py
│   │   ├── schema_check.py      # V1: Schema validation
│   │   ├── join_check.py        # V3: Join & leakage validation
│   │   └── metrics_check.py     # V4: Metrics & transparency validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_utils.py     # Colored logging, file handlers
│       ├── file_utils.py        # CSV/JSON/Parquet I/O
│       └── stats_utils.py       # Type inference, statistics, outliers
│
├── docs/
│   ├── ARCHITECTURE.md          # System design, data flow, extension points
│   └── STAGES.md                # Stage-by-stage examples and guides
│
├── tests/                       # Unit and integration tests
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_stage4.py
│
└── logs/                        # Pipeline logs (auto-created)
    └── pipeline.log
```

## Key Features

### 1. Intelligent Data Understanding (Stage 1)

- **Type Inference:** Automatically detects numeric, categorical, date, string columns
- **Key Detection:** Identifies primary keys (>95% unique) and foreign keys (common join columns)
- **Composite Keys:** Finds multi-column keys (e.g., state+year+crop)
- **Data Quality:** Reports null rates, outliers, value distributions

### 2. Smart Task Suggestion (Stage 2)

- **Prediction Tasks:** Finds numeric targets with time structure and joinable features
- **Descriptive Tasks:** Identifies aggregation opportunities (mean by state, top producers)
- **Clustering Tasks:** Detects entity columns with multiple numeric features
- **Feasibility Scoring:** Ranks tasks by data quality and join potential

### 3. Robust Data Preparation (Stage 3)

- **Key Normalization:** Maps variants to standard forms ("UP" → "Uttar Pradesh")
- **Smart Joins:** Left/inner joins with cardinality tracking
- **Feature Engineering:**
  - Lag features (t-1, t-2 for time series)
  - Rolling statistics (3-year, 5-year averages)
  - Anomaly detection (deviation from group means)
- **Data Cleaning:** Drops rows with excessive missing values, imputes remaining

### 4. Production-Ready Modeling (Stage 4)

- **Multiple Models:**
  - Ridge Regression (interpretable baseline)
  - XGBoost (high-performance gradient boosting)
  - Random Forest (optional, for comparison)
- **Proper Splits:** Time-based (≤2020 vs >2020) or random (80/20)
- **Cross-Validation:** 5-fold CV on training set
- **Rich Outputs:**
  - 5 types of plots (actual vs pred, residuals, time series, feature importance, errors)
  - Model cards with full feature lists and hyperparameters
  - Predictions CSV with residuals for error analysis

### 5. Comprehensive Verification

- **V1 (Schema):** Column type agreement, range sanity, null thresholds
- **V2 (Human):** Interactive task selection
- **V3 (Joins):** Join cardinality, coverage, leakage detection
- **V4 (Metrics):** R² thresholds, residual diagnostics, model transparency

## Usage Examples

### Quick Start (One Command)

```bash
# 1. Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add data
cp ~/crop_yield.csv data/raw/
cp ~/rainfall.csv data/raw/

# 3. Run pipeline
python src/main.py --mode full

# 4. View outputs
ls data/outputs/
```

### Stage-by-Stage

```bash
# Analyze data
python src/main.py --mode stage1
# → Check data/summaries/*.json

# Get task suggestions
python src/main.py --mode stage2
# → Check data/tasks.json

# Prepare data for task T1
python src/main.py --mode stage3 --task-id T1
# → Check data/intermediate/T1_merged.parquet

# Train models for task T1
python src/main.py --mode stage4 --task-id T1
# → Check data/outputs/T1_*.json, data/outputs/plots/*.png
```

### Configuration

```yaml
# config/pipeline_config.yaml

# Stage 1: Control sampling
summarizer:
  sample_size: 1000  # null = all rows

# Stage 3: Configure features
planner:
  features:
    create_lags: true
    lag_periods: [1, 2, 3]  # t-1, t-2, t-3

# Stage 4: Tune models
executor:
  split:
    test_split_year: 2020
  models:
    gradient_boosting:
      n_estimators: 200  # more trees
      max_depth: 8       # deeper trees
      learning_rate: 0.05  # slower learning
```

## Sample Output

### Metrics (T1_metrics.json)
```json
{
  "task_id": "T1",
  "best_model": "gradient_boosting",
  "models": {
    "gradient_boosting": {
      "test": {
        "mae": 220.1,
        "rmse": 280.3,
        "r2": 0.78,
        "mape": 12.5
      }
    }
  }
}
```

### Predictions (T1_predictions.csv)
```csv
state,year,crop,actual,predicted,residual
Punjab,2021,Rice,3548,3520,-28
Haryana,2021,Wheat,3620,3680,+60
...
```

### Visualizations
- `actual_vs_pred_gradient_boosting.png` - Scatter plot
- `residuals_by_state_gradient_boosting.png` - Bar chart of errors by state
- `time_series_gradient_boosting.png` - Actual vs predicted over time
- `feature_importance_gradient_boosting.png` - Top 20 features
- `error_distribution_gradient_boosting.png` - Histogram + Q-Q plot

## Performance

### Benchmarks (on 15,000-row dataset)

| Stage | Time | Output |
|-------|------|--------|
| Stage 1 | 5 sec | 3 summaries |
| Stage 2 | 2 sec | 5 task suggestions |
| Stage 3 | 8 sec | 10,000-row merged dataset |
| Stage 4 | 30 sec | 2 models, 5 plots |
| **Total** | **~45 sec** | Complete analysis |

### Scalability

- **Small datasets (<1K rows):** No optimization needed
- **Medium datasets (1K-100K):** Enable sampling in Stage 1
- **Large datasets (>100K):** Use Parquet, parallel processing (future)

## Future Enhancements

1. **Web UI:** Flask dashboard for task selection and result viewing
2. **AutoML:** Automated hyperparameter tuning with Optuna
3. **Model Registry:** MLflow integration for versioning
4. **Streaming:** Real-time updates for incremental data
5. **Distributed:** Celery/Dask for parallel task execution

## Academic Context

**Course:** Capstone Project
**Focus:** Agentic AI Systems for Data Analysis
**Key Concepts:**
- Multi-agent architectures
- Verification-driven development
- Reproducible ML pipelines
- Agricultural data analysis

## Getting Help

- **Setup issues:** See `README.md` → Troubleshooting
- **Usage questions:** See `QUICKSTART.md`
- **Architecture details:** See `docs/ARCHITECTURE.md`
- **Stage examples:** See `docs/STAGES.md`
- **Logs:** Check `logs/pipeline.log`

## License

Educational use only. Not for commercial distribution.

---

**Built with:** Python 3.10, pandas, scikit-learn, XGBoost, matplotlib
**Author:** Capstone Project Team
**Date:** January 2025
