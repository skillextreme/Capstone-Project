# System Architecture

## Overview

The Agentic Pipeline is a 4-stage system for analyzing agricultural datasets with automated verification at each checkpoint. This document explains the system design, data flow, and key design decisions.

## High-Level Architecture

```
┌─────────────────────┐
│   Raw Data Files    │
│  (CSV/JSON)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    STAGE 1:         │
│    Summarizer       │  ← Analyzes schema, stats, keys
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  VERIFICATION V1:   │
│  Schema Check       │  ← Validates summaries
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    STAGE 2:         │
│  Task Suggester     │  ← Proposes feasible analyses
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  VERIFICATION V2:   │
│  Human Selection    │  ← User picks task
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    STAGE 3:         │
│    Planner          │  ← Joins data, engineers features
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  VERIFICATION V3:   │
│  Join & Leakage     │  ← Validates joins
│  Check              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    STAGE 4:         │
│    Executor         │  ← Trains models, generates outputs
│                     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  VERIFICATION V4:   │
│  Metrics Check      │  ← Validates model quality
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Final Outputs:     │
│  - Predictions      │
│  - Metrics          │
│  - Visualizations   │
│  - Model Cards      │
└─────────────────────┘
```

## Design Principles

### 1. Statelessness

Each stage is **stateless** - they don't maintain internal state between runs. All intermediate results are written to disk with full provenance information.

**Benefits:**
- Easy to debug (inspect artifacts at any stage)
- Resilient (can resume from any checkpoint)
- Reproducible (same inputs → same outputs)

**Implementation:**
- Stage 1 writes JSON summaries to `data/summaries/`
- Stage 2 writes task proposals to `data/tasks.json`
- Stage 3 writes merged data and plans to `data/intermediate/`
- Stage 4 writes all outputs to `data/outputs/`

### 2. Verification Gates

Each stage has an automated verification checkpoint:

| Verification | Purpose | Key Checks |
|--------------|---------|------------|
| V1 | Schema validation | Column types, ranges, null rates |
| V2 | Human adjudication | Task selection |
| V3 | Join validation | Cardinality, coverage, leakage |
| V4 | Model validation | Metrics, residuals, transparency |

**Non-blocking:** Verifications report issues but don't halt the pipeline (configurable).

### 3. Modularity

Each stage can run independently:

```bash
# Run just Stage 1
python src/main.py --mode stage1

# Run just Stage 4 for a specific task
python src/main.py --mode stage4 --task-id T1
```

This enables:
- Iterative development
- Targeted debugging
- Partial re-runs

## Data Flow

### Stage 1: Summarizer → V1

**Input:** Raw CSV/JSON files in `data/raw/`

**Process:**
1. Load file (with optional sampling for large files)
2. For each column:
   - Infer type (numeric, categorical, date, string, boolean)
   - Calculate statistics (mean, std, cardinality, nulls)
   - Extract sample values
3. Detect candidate keys (primary and foreign)
4. Save summary to JSON

**Output:** `data/summaries/<file>.summary.json`

**Verification V1:**
- Loads original file and summary
- Checks column names match
- Verifies type agreement (>95% of values parse as declared type)
- Validates ranges (e.g., years in [1900, 2030])
- Flags high null rates (>95%)

**Key Code:**
- `src/stage1/summarizer.py` - Main summarizer
- `src/verifiers/schema_check.py` - V1 verification
- `src/utils/stats_utils.py` - Type inference, stats calculation

---

### Stage 2: Task Suggester → V2

**Input:** All summary JSONs from Stage 1

**Process:**
1. Load all summaries
2. Apply rule-based heuristics:
   - **Prediction tasks:** Find numeric targets + time structure + joinable features
   - **Descriptive tasks:** Find aggregatable columns + grouping keys
   - **Clustering tasks:** Find multiple numeric features + entity columns
3. Score feasibility based on:
   - Key overlap between files
   - Data quality (low null rates)
   - Sufficient rows
4. Generate task proposals (JSON format)

**Output:** `data/tasks.json`

**Verification V2 (Human Adjudication):**
- Display tasks to user
- User selects which task(s) to execute
- For automation: auto-select high-feasibility tasks

**Key Code:**
- `src/stage2/task_suggester.py` - Task proposal logic

---

### Stage 3: Planner → V3

**Input:**
- Selected task from V2
- Raw data files specified in task

**Process:**
1. **Load & Normalize:**
   - Load all required files
   - Normalize key columns (e.g., "UP" → "Uttar Pradesh")
2. **Merge:**
   - Determine join keys (from task or infer)
   - Progressively merge files using pandas
   - Track row counts at each join
3. **Feature Engineering:**
   - Create lag features (e.g., previous year's yield)
   - Create rolling statistics (3-year, 5-year averages)
   - Optionally create interactions
4. **Data Cleaning:**
   - Drop rows with >50% missing values
   - Handle outliers (cap or remove based on config)
5. **Save:**
   - Merged data → `data/intermediate/<task_id>_merged.parquet`
   - Join plan → `data/intermediate/<task_id>_plan.json`

**Output:**
- `data/intermediate/<task_id>_merged.parquet`
- `data/intermediate/<task_id>_plan.json`

**Verification V3:**
- **Join cardinality:** Check for duplicate join keys (flag many-to-many)
- **Coverage:** Check for excessive missing values after join
- **Leakage:** For time-series, verify lag features don't have values in early years

**Key Code:**
- `src/stage3/planner.py` - Join orchestration, feature engineering
- `src/stage3/normalizer.py` - Key normalization
- `src/verifiers/join_check.py` - V3 verification

---

### Stage 4: Executor → V4

**Input:**
- Task definition
- Merged data from Stage 3

**Process:**
1. **Prepare Features:**
   - Extract numeric features (drop IDs like state, crop)
   - Extract target variable
   - Handle missing values (median imputation)
2. **Split Data:**
   - Time-based split: train on ≤2020, test on >2020
   - Or random split: 80/20
3. **Train Models:**
   - Linear Regression (Ridge)
   - Gradient Boosting (XGBoost)
   - Cross-validation on training set
4. **Evaluate:**
   - Calculate metrics: MAE, RMSE, R², MAPE
   - Generate predictions on test set
   - Compute residuals
5. **Visualize:**
   - Actual vs Predicted scatter plot
   - Residuals by segment (state, crop)
   - Time series plots
   - Feature importance
   - Error distribution (histogram + Q-Q plot)
6. **Save Outputs:**
   - Metrics → `data/outputs/<task_id>_metrics.json`
   - Predictions → `data/outputs/<task_id>_predictions.csv`
   - Model card → `data/outputs/<task_id>_model_card.json`
   - Plots → `data/outputs/plots/`

**Output:**
- Metrics, predictions, model card, plots

**Verification V4:**
- **Metrics:** Check R² > threshold, MAPE < threshold, flag suspiciously high R² (>0.99)
- **Residuals:** Check mean ≈ 0, skewness < 2
- **Transparency:** Verify model card has all required fields

**Key Code:**
- `src/stage4/executor.py` - Main orchestration
- `src/stage4/models.py` - Model training (Ridge, XGBoost, RF)
- `src/stage4/visualizer.py` - All plots
- `src/verifiers/metrics_check.py` - V4 verification

---

## Configuration System

### YAML Config (`config/pipeline_config.yaml`)

Centralized configuration for all stages:

```yaml
data:
  raw_dir: "data/raw"
  summaries_dir: "data/summaries"
  # ...

summarizer:
  sample_size: 1000
  numeric_threshold: 0.95
  # ...

executor:
  split:
    method: "time"
    test_split_year: 2020
  models:
    linear_regression:
      enabled: true
      regularization: "ridge"
    # ...
```

### Environment Variables (`.env`)

Optional overrides:

```bash
OPENAI_API_KEY=...
TEST_SPLIT_YEAR=2020
LOG_LEVEL=INFO
```

### Config Class (`src/config.py`)

Loads YAML + env vars, provides dot-notation access:

```python
config = get_config()
raw_dir = config.get('data.raw_dir')  # "data/raw"
```

---

## Utilities

### Logging (`src/utils/logging_utils.py`)

- Colored console output
- File logging with rotation
- Per-module loggers

### File I/O (`src/utils/file_utils.py`)

- Unified CSV/JSON/Parquet loading
- Config loading from YAML
- Automatic directory creation

### Statistics (`src/utils/stats_utils.py`)

- Type inference (numeric vs categorical vs date)
- Basic statistics (mean, std, cardinality)
- Outlier detection (IQR, z-score)
- Key candidacy checks

---

## Extension Points

### Adding a New Model

1. Add config to `config/pipeline_config.yaml`:
   ```yaml
   executor:
     models:
       my_new_model:
         enabled: true
         param1: value1
   ```

2. Implement trainer in `src/stage4/models.py`:
   ```python
   def _train_my_new_model(self, X_train, y_train, X_test, y_test):
       model = MyModel(**self.config['models']['my_new_model'])
       model.fit(X_train, y_train)
       # ...
       return results
   ```

3. Call from `ModelTrainer.train_and_evaluate()`

### Adding a New Verification

1. Create `src/verifiers/my_check.py`
2. Implement checker class
3. Call from `src/main.py` pipeline

### Adding LLM-Powered Task Suggestion

1. Set `task_suggester.llm.provider: "openai"` in config
2. Add `OPENAI_API_KEY` to `.env`
3. Implement `_suggest_with_llm()` in `task_suggester.py`

---

## Error Handling

- **File not found:** Logged as error, stage skips that file
- **Verification failures:** Logged but don't block pipeline (configurable)
- **Missing dependencies:** Raises clear error message

---

## Performance Considerations

- **Sampling:** Stage 1 can sample large files (set `sample_size`)
- **Parallelization:** Stages are sequential, but multiple tasks can run in parallel
- **Memory:** Parquet format for intermediate storage (efficient compression)
- **Caching:** Summaries are cached; re-running Stage 2-4 doesn't re-analyze raw files

---

## Security

- **API keys:** Stored in `.env` (gitignored)
- **Data privacy:** All processing is local (no external API calls for data)
- **Input validation:** File types checked before loading

---

## Testing Strategy

Unit tests for each module:

```bash
pytest tests/test_stage1.py -v
pytest tests/test_stage2.py -v
# ...
```

Integration test for full pipeline:

```bash
pytest tests/test_pipeline.py -v
```

---

## Future Enhancements

1. **Web UI:** Flask/FastAPI dashboard for task selection and result viewing
2. **Distributed Execution:** Celery for parallel task execution
3. **Model Registry:** MLflow for model versioning
4. **AutoML:** Hyperparameter tuning with Optuna
5. **Real-time Updates:** Stream processing for incremental data

---

## References

- Pandas: Data manipulation
- Scikit-learn: Machine learning
- XGBoost: Gradient boosting
- Matplotlib/Seaborn: Visualization
- PyYAML: Configuration
