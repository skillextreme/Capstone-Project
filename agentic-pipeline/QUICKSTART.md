# Quick Start Guide

Get up and running with the Agentic Pipeline in 5 minutes.

## Prerequisites

- Python 3.8+
- 4GB RAM minimum
- Basic familiarity with command line

## Installation (5 steps)

### 1. Navigate to project directory

```bash
cd "/home/jacob-michael-mathew/Desktop/Capstone Project/Capstone-Project/agentic-pipeline"
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data processing)
- scikit-learn, xgboost (machine learning)
- matplotlib, seaborn (visualization)
- pyyaml, python-dotenv (configuration)

### 4. Add your data

Place CSV files in `data/raw/`:

```bash
# Example: Copy your agricultural data files
cp ~/path/to/crop_yield.csv data/raw/
cp ~/path/to/rainfall.csv data/raw/
cp ~/path/to/fertilizer_usage.csv data/raw/
```

**File format requirements:**
- CSV or JSON format
- Column headers in first row
- Common key columns across files (e.g., state, year, crop)

### 5. Run the pipeline

```bash
python src/main.py --mode full
```

That's it! The pipeline will:
1. Analyze your data files (Stage 1)
2. Suggest analysis tasks (Stage 2)
3. Join data and engineer features (Stage 3)
4. Train models and generate outputs (Stage 4)

## What You Get

After running, check `data/outputs/`:

```
data/outputs/
├── T1_metrics.json              # Model performance metrics
├── T1_predictions.csv           # Predictions with actuals
├── T1_model_card.json           # Model documentation
└── plots/
    ├── actual_vs_pred_gradient_boosting.png
    ├── residuals_by_state_gradient_boosting.png
    ├── time_series_gradient_boosting.png
    ├── feature_importance_gradient_boosting.png
    └── error_distribution_gradient_boosting.png
```

## Example with Sample Data

Don't have data yet? Create a sample dataset:

```bash
# Create sample crop yield data
python << 'EOF'
import pandas as pd
import numpy as np

np.random.seed(42)

states = ['Punjab', 'Haryana', 'UP', 'Bihar', 'Maharashtra']
crops = ['Rice', 'Wheat', 'Sugarcane']
years = range(2010, 2024)

data = []
for state in states:
    for crop in crops:
        for year in years:
            data.append({
                'state': state,
                'year': year,
                'crop': crop,
                'area_hectares': np.random.randint(50000, 500000),
                'production_tonnes': np.random.randint(100000, 2000000),
                'yield_kg_per_hectare': np.random.randint(2000, 5000)
            })

df = pd.DataFrame(data)
df.to_csv('data/raw/crop_yield.csv', index=False)
print(f"Created sample data: {len(df)} rows")
EOF

# Run pipeline on sample data
python src/main.py --mode full
```

## Running Individual Stages

### Stage 1 only: Analyze data

```bash
python src/main.py --mode stage1
```

Check `data/summaries/` for JSON summaries of each file.

### Stage 2 only: Get task suggestions

```bash
python src/main.py --mode stage2
```

Check `data/tasks.json` for proposed analyses.

### Stage 3 only: Prepare data for a task

```bash
python src/main.py --mode stage3 --task-id T1
```

Check `data/intermediate/T1_merged.parquet` for joined data.

### Stage 4 only: Train models

```bash
python src/main.py --mode stage4 --task-id T1
```

Check `data/outputs/` for results.

## Configuration

### Basic settings

Edit `config/pipeline_config.yaml`:

```yaml
# Change test/train split year
executor:
  split:
    test_split_year: 2020  # Train on ≤2020, test on >2020

# Enable/disable models
executor:
  models:
    linear_regression:
      enabled: true
    gradient_boosting:
      enabled: true
    random_forest:
      enabled: false  # Disable RF
```

### Logging

```yaml
logging:
  level: "INFO"  # DEBUG for more detail
  console:
    enabled: true
    colorize: true
```

## Troubleshooting

### "No files found in data/raw/"

**Solution:** Add CSV files to `data/raw/` directory.

### "Module not found" errors

**Solution:** Activate virtual environment:
```bash
source venv/bin/activate
```

### "Memory error"

**Solution:** Reduce sample size in config:
```yaml
summarizer:
  sample_size: 1000  # Analyze only 1000 rows
```

### "Join explosion" warning

**Solution:** Check for duplicate keys in source files. The pipeline will warn if joins create many-to-many relationships.

## Next Steps

1. **Explore outputs:** Open plots in `data/outputs/plots/`
2. **Review metrics:** Check `T1_metrics.json` for model performance
3. **Customize:** Edit `config/pipeline_config.yaml` to tune models
4. **Read docs:** See `README.md` for full documentation

## Common Use Cases

### Predict crop yields

Files needed:
- `crop_yield.csv` (state, year, crop, yield)
- `rainfall.csv` (state, year, rainfall_mm)
- `fertilizer_usage.csv` (state, year, fertilizer_kg)

The pipeline will:
1. Auto-detect that yield is a prediction target
2. Join rainfall and fertilizer data
3. Create lag features (previous year's yield)
4. Train models and report accuracy

### Find top producing states

File needed:
- `crop_production.csv` (state, year, crop, production)

The pipeline will:
1. Suggest a descriptive task: "Find top states by production"
2. Group by state and calculate means
3. Generate ranking tables

### Cluster states by crop profiles

File needed:
- `crop_yield.csv` (state, year, crop1_yield, crop2_yield, ...)

The pipeline will:
1. Suggest clustering task
2. Normalize features
3. Run K-means clustering
4. Report cluster memberships

## Help & Support

- **Documentation:** `README.md`, `docs/ARCHITECTURE.md`, `docs/STAGES.md`
- **Examples:** See `docs/STAGES.md` for detailed examples
- **Issues:** Check logs in `logs/pipeline.log`

## Performance Tips

- Use `--verbose` flag for detailed logs: `python src/main.py --mode full --verbose`
- For large datasets, enable sampling in Stage 1
- Run stages independently for faster iteration
- Use Parquet format for large intermediate files (already default)

---

**You're all set! Happy analyzing!**
