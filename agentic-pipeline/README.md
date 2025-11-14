# Agricultural Data Analysis: 4-Stage Agentic Pipeline

## Table of Contents
1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Project Structure](#project-structure)
4. [Pipeline Stages](#pipeline-stages)
5. [Usage Guide](#usage-guide)
6. [Configuration](#configuration)
7. [Verification System](#verification-system)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **4-stage agentic pipeline** for analyzing agricultural datasets (crop yields, rainfall, inputs, etc.). The pipeline is designed to be:
- **Reproducible**: All stages write artifacts to disk with full traceability
- **Verifiable**: Each stage has automated verification checkpoints
- **Stateless**: Stages don't maintain internal state but persist results
- **Modular**: Each stage can be run independently or as part of the full pipeline

### What This System Does

1. **Stage 1 (Summarizer)**: Analyzes raw CSV/JSON files to extract schema, statistics, and candidate keys
2. **Stage 2 (Task Suggester)**: Proposes feasible analyses based on available data
3. **Stage 3 (Planner)**: Creates reproducible data plans with joins, feature engineering, and normalization
4. **Stage 4 (Executor)**: Runs analyses and generates predictions, visualizations, and reports

---

## Setup Instructions

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: At least 4GB (8GB+ recommended for large datasets)
- **Disk Space**: 1GB for dependencies + space for your data

### Step 1: Clone or Navigate to Project

```bash
cd "/home/jacob-michael-mathew/Desktop/Capstone Project/Capstone-Project/agentic-pipeline"
```

### Step 2: Create Virtual Environment

**Using venv (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate  # On Windows
```

**Using conda:**
```bash
conda create -n agri-pipeline python=3.10
conda activate agri-pipeline
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting
- `matplotlib`, `seaborn` - Visualization
- `openai` - LLM integration for agents (optional)
- `anthropic` - Claude API for agents (optional)

### Step 4: Configure API Keys (Optional)

If you want to use LLM-powered agents for intelligent suggestions:

```bash
# Create .env file
cp config/.env.example .env

# Edit .env and add your API keys
nano .env
```

Add:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Note**: The pipeline works without API keys using rule-based heuristics.

### Step 5: Verify Installation

```bash
python -m pytest tests/ -v
```

### Step 6: Add Sample Data

Place your CSV/JSON files in the `data/raw/` directory:

```bash
# Example structure:
data/raw/
  ├── crop_yield.csv
  ├── rainfall.csv
  └── inputs.csv
```

### Step 7: Run Your First Pipeline

```bash
# Run the complete pipeline
python src/main.py --mode full

# Or run individual stages
python src/main.py --mode stage1  # Summarize only
```

---

## Project Structure

```
agentic-pipeline/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── .env.example              # Environment variables template
│
├── data/                     # Data storage (NOT committed to git)
│   ├── raw/                  # Original CSV/JSON files
│   ├── summaries/            # Stage 1 outputs (JSON summaries)
│   ├── intermediate/         # Stage 3 outputs (cleaned, joined data)
│   └── outputs/              # Stage 4 outputs (models, plots, reports)
│
├── src/                      # Source code
│   ├── main.py               # Pipeline orchestrator
│   ├── config.py             # Global configuration
│   │
│   ├── stage1/               # Stage 1: Summarizer
│   │   ├── __init__.py
│   │   └── summarizer.py     # File analysis and summary generation
│   │
│   ├── stage2/               # Stage 2: Task Suggester
│   │   ├── __init__.py
│   │   └── task_suggester.py # Analysis task proposals
│   │
│   ├── stage3/               # Stage 3: Planner
│   │   ├── __init__.py
│   │   ├── planner.py        # Join planning and feature engineering
│   │   └── normalizer.py     # Key normalization
│   │
│   ├── stage4/               # Stage 4: Executor
│   │   ├── __init__.py
│   │   ├── executor.py       # Run analyses
│   │   ├── models.py         # ML models
│   │   └── visualizer.py     # Plots and charts
│   │
│   ├── verifiers/            # Verification checkpoints
│   │   ├── __init__.py
│   │   ├── schema_check.py   # V1: Schema validation
│   │   ├── join_check.py     # V3: Join cardinality validation
│   │   └── metrics_check.py  # V4: Model metrics validation
│   │
│   └── utils/                # Shared utilities
│       ├── __init__.py
│       ├── file_utils.py     # File I/O helpers
│       ├── logging_utils.py  # Logging configuration
│       └── stats_utils.py    # Statistical helpers
│
├── config/                   # Configuration files
│   ├── pipeline_config.yaml  # Pipeline settings
│   └── .env.example          # API keys template
│
├── tests/                    # Unit and integration tests
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_stage4.py
│
└── docs/                     # Detailed documentation
    ├── ARCHITECTURE.md       # System design
    ├── STAGES.md             # Stage-by-stage guide
    └── API.md                # API reference
```

---

## Pipeline Stages

### Stage 1: Summarizer Agent

**Purpose**: Generate factual summaries of each CSV/JSON file.

**Inputs**:
- Raw data files from `data/raw/`

**Outputs** (saved to `data/summaries/`):
```json
{
  "file_name": "crop_yield.csv",
  "columns": [
    {
      "name": "state",
      "type": "categorical",
      "null_rate": 0.02,
      "cardinality": 35,
      "sample_values": ["Punjab", "Haryana", "UP"]
    },
    {
      "name": "year",
      "type": "numeric",
      "null_rate": 0.0,
      "min": 2000,
      "max": 2023,
      "mean": 2011.5
    }
  ],
  "candidate_keys": {
    "primary": ["state", "year", "crop"],
    "foreign": ["state", "year"]
  },
  "row_count": 15000
}
```

**Verification V1**: Schema checker validates column types, ranges, and key candidates.

---

### Stage 2: Task Suggestion Agent

**Purpose**: Propose feasible analyses based on available data summaries.

**Inputs**:
- All JSON summaries from Stage 1

**Outputs** (interactive or saved to `data/tasks.json`):
```json
[
  {
    "task_id": "T1",
    "type": "prediction",
    "description": "Predict next-season yield by state and crop",
    "target_variable": "yield",
    "features": ["rainfall", "fertilizer_usage", "previous_yield"],
    "required_files": ["crop_yield.csv", "rainfall.csv", "inputs.csv"],
    "required_keys": ["state", "year", "crop"]
  },
  {
    "task_id": "T2",
    "type": "descriptive",
    "description": "Find largest producing state per crop per year",
    "aggregation": "max",
    "group_by": ["year", "crop"]
  }
]
```

**Verification V2**: Human adjudication - user selects which task to run.

---

### Stage 3: Planner / Join Builder

**Purpose**: Create a reproducible data plan for the selected task.

**Steps**:
1. Normalize key names (map "UP" → "Uttar Pradesh")
2. Build join graph (yield ⋊⋉ rainfall ⋊⋉ inputs on state×year)
3. Engineer features (lagged yield, rainfall anomalies, per-capita rates)
4. Save intermediate table to `data/intermediate/`

**Outputs**:
- `data/intermediate/merged_data.parquet` - Cleaned, joined dataset
- `data/intermediate/join_plan.json` - Provenance and join details
- `data/intermediate/features.json` - Feature engineering log

**Verification V3**:
- Join cardinality checks (flag many-to-many explosions)
- Coverage checks (% rows retained after joins)
- Leakage checks (ensure no future data in training set)

---

### Stage 4: Executor

**Purpose**: Run the analysis and produce outputs.

**For Prediction Tasks**:
1. Split data by time (train on years ≤ 2020, test on 2021-2023)
2. Fit baseline models:
   - Regularized Linear Regression
   - Gradient Boosting (XGBoost)
3. Generate predictions and residuals
4. Create visualizations (actual vs. predicted, error by segment)

**For Descriptive Tasks**:
1. Execute groupby/aggregation queries
2. Return tidy tables

**Outputs** (saved to `data/outputs/`):
```
outputs/
├── metrics.json              # MAE, RMSE, R², MAPE
├── predictions.csv           # state, year, crop, actual, predicted, residual
├── model_card.json           # Model config, features used, timestamp
├── plots/
│   ├── actual_vs_pred.png
│   ├── residuals_by_state.png
│   └── residuals_by_crop.png
└── report.html               # Auto-generated summary report
```

**Verification V4**:
- Report metrics on holdout set
- Check residual mean ≈ 0
- Verify no data leakage (future years excluded from training)

---

## Usage Guide

### Basic Usage

```bash
# Run full pipeline (all 4 stages)
python src/main.py --mode full

# Run specific stage
python src/main.py --mode stage1
python src/main.py --mode stage2
python src/main.py --mode stage3 --task-id T1
python src/main.py --mode stage4 --task-id T1

# Run with verbose logging
python src/main.py --mode full --verbose

# Use specific LLM (if API keys configured)
python src/main.py --mode full --llm anthropic
```

### Python API Usage

```python
from src.stage1.summarizer import Summarizer
from src.stage2.task_suggester import TaskSuggester
from src.stage3.planner import Planner
from src.stage4.executor import Executor

# Stage 1: Summarize data
summarizer = Summarizer(data_dir="data/raw", output_dir="data/summaries")
summaries = summarizer.run_all()

# Stage 2: Get task suggestions
suggester = TaskSuggester(summaries_dir="data/summaries")
tasks = suggester.suggest_tasks()

# Stage 3: Plan for a specific task
planner = Planner(task=tasks[0], summaries=summaries)
plan = planner.create_plan()
merged_data = planner.execute_plan()

# Stage 4: Execute analysis
executor = Executor(data=merged_data, task=tasks[0])
results = executor.run()
```

---

## Configuration

### Pipeline Settings (`config/pipeline_config.yaml`)

```yaml
# Data paths
data:
  raw_dir: "data/raw"
  summaries_dir: "data/summaries"
  intermediate_dir: "data/intermediate"
  outputs_dir: "data/outputs"

# Stage 1: Summarizer
summarizer:
  sample_size: 1000  # Rows to sample for type inference
  null_threshold: 0.95  # Flag columns with >95% nulls
  numeric_threshold: 0.95  # Treat as numeric if ≥95% rows parse

# Stage 3: Planner
planner:
  join_type: "left"  # Default join strategy
  drop_na_threshold: 0.5  # Drop rows with >50% missing after join

# Stage 4: Executor
executor:
  test_split_year: 2020  # Train on ≤2020, test on >2020
  models:
    - linear_regression
    - xgboost
  cv_folds: 5
  random_seed: 42

# Verification
verification:
  v1_enabled: true
  v3_enabled: true
  v4_enabled: true
  fail_on_error: false  # Continue even if verification fails
```

---

## Verification System

### V1: Schema Check (after Stage 1)

Validates:
- Column names match expected schema
- Data types are correctly inferred
- Ranges are plausible (e.g., yields > 0, years in [1900, 2025])
- Null rates are acceptable

### V2: Human Adjudication (after Stage 2)

User reviews proposed tasks and selects one to execute.

### V3: Join & Leakage Check (after Stage 3)

Validates:
- Join cardinality (1:1, 1:many, or many:many)
- Row coverage (% of rows retained after joins)
- No data leakage (train/test split by time)

### V4: Metrics Check (after Stage 4)

Reports:
- Holdout metrics (MAE, RMSE, R², MAPE)
- Residual diagnostics (mean, std, distribution)
- Model transparency (features used, hyperparameters)

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "No files found in data/raw/"

**Solution**: Add your CSV/JSON files to the `data/raw/` directory.

### Issue: API key errors

**Solution**: Either:
1. Add valid API keys to `.env`
2. OR run without LLM features (pipeline will use rule-based heuristics)

### Issue: Out of memory errors

**Solution**:
- Reduce sample size in `config/pipeline_config.yaml`
- Process files one at a time
- Use a machine with more RAM

### Issue: Join explosion (too many rows after join)

**Solution**: Check Stage 3 logs for many-to-many warnings. You may need to:
- Deduplicate input files
- Adjust join keys
- Use different join type (inner vs. left)

---

## Contributing

This is a capstone project. For questions or issues, contact the project maintainer.

---

## License

Educational use only. Not for commercial distribution.
