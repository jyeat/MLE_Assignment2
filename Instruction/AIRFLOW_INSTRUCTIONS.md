# Complete Airflow Backfill Instructions

## Understanding the Two-Phase Approach

### Phase 1: Initial Backfill (Data Pipeline + Model Training)
- Runs: 2023-01-01 to 2025-01-01 (25 months)
- What happens:
  - Months 2023-01 to 2024-12: Data pipeline only (training/inference skipped)
  - Month 2025-01: Data pipeline + MODEL TRAINING + inference for 2025-01
- Result: Model trained and stored in /app/models/, 1 prediction file

### Phase 2: Historical Inference (Monitoring Across Time)
- Re-runs inference tasks for ALL months
- Now that model exists, inference works for all historical data
- Result: 25 prediction files (one per month) for performance monitoring

---

## Step-by-Step Commands

### 1. Initialize Airflow (First Time Only)

```bash
export AIRFLOW_HOME=/app

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 2. Start Airflow Services

```bash
# Start webserver
airflow webserver --port 8080 -D

# Start scheduler
airflow scheduler -D

# Wait for services to start
sleep 10

# Verify DAG is loaded
airflow dags list | grep data_pipeline_dag
```

### 3. Phase 1 - Initial Backfill (Data + Training)

```bash
# Run complete backfill from start to end
airflow dags backfill \
    --start-date 2023-01-01 \
    --end-date 2025-01-01 \
    --reset-dagruns \
    data_pipeline_dag
```

**Expected output:**
- 25 DAG runs (one per month)
- Months 1-24: Data processing only
- Month 25 (2025-01): Data processing + **MODEL TRAINING** + inference

**Wait for completion** (this may take 4-7 hours)

### 4. Verify Model Training Succeeded

```bash
# Check model bank
ls -la /app/models/

# Should see:
# - xgboost_credit_default_model.pkl
# - preprocessors.pkl
# - model_metadata.pkl
# - feature_importance.csv

# Check initial predictions (only 2025-01)
ls -la /app/predictions/

# Should see: predictions_2025-01-01_*.csv
```

### 5. Phase 2 - Historical Inference (All Months)

Now that the model exists, re-run inference for ALL historical months:

```bash
# Clear and re-run ONLY inference tasks for all months
airflow tasks clear \
    --yes \
    --only-failed false \
    --task-regex "run_model_inference" \
    --start-date 2023-01-01 \
    --end-date 2025-01-01 \
    data_pipeline_dag

# Alternative: Use backfill with --rerun-failed-tasks
airflow dags backfill \
    --start-date 2023-01-01 \
    --end-date 2025-01-01 \
    --rerun-failed-tasks \
    --task-regex "run_model_inference" \
    data_pipeline_dag
```

**This will:**
- Re-run inference for all 25 months
- Model now exists, so inference succeeds for all months
- Generate 25 prediction files

### 6. Verify Complete Output

```bash
# Check all predictions (should have 25 files)
ls -la /app/predictions/
wc -l /app/predictions/*.csv

# View sample predictions
head -10 /app/predictions/predictions_2023-01-01_*.csv
head -10 /app/predictions/predictions_2025-01-01_*.csv
```

---

## Monitoring Model Performance (Requirement c)

After both phases complete, you'll have:

**1. Model Artifacts** (`/app/models/`)
- Trained model
- Training metrics (test AUC, OOT AUC)
- Feature importance

**2. Monthly Predictions** (`/app/predictions/`)
- predictions_2023-01-01_*.csv
- predictions_2023-02-01_*.csv
- ... (25 total files)
- predictions_2025-01-01_*.csv

**3. Monitor Over Time:**

```python
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Load all predictions
pred_files = sorted(glob.glob('/app/predictions/predictions_*.csv'))
print(f"Found {len(pred_files)} prediction files")

# Analyze monthly trends
monthly_stats = []
for f in pred_files:
    df = pd.read_csv(f)
    monthly_stats.append({
        'month': df['inference_date'].iloc[0],
        'n_customers': len(df),
        'default_rate': (df['predicted_default'] == 1).mean(),
        'avg_probability': df['default_probability'].mean(),
        'high_risk_pct': (df['risk_category'] == 'High Risk').mean()
    })

stats_df = pd.DataFrame(monthly_stats)
print(stats_df)

# Plot trends
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

stats_df.plot(x='month', y='default_rate', ax=axes[0,0], title='Default Rate Over Time')
stats_df.plot(x='month', y='avg_probability', ax=axes[0,1], title='Avg Default Probability')
stats_df.plot(x='month', y='high_risk_pct', ax=axes[1,0], title='High Risk % Over Time')
stats_df.plot(x='month', y='n_customers', ax=axes[1,1], title='Customers Scored')

plt.tight_layout()
plt.savefig('/app/model_monitoring_report.png')
print("Monitoring report saved to /app/model_monitoring_report.png")
```

---

## Timeline Estimate

**Phase 1 (Initial Backfill):**
- 24 runs × 10-15 min = 4-6 hours
- 1 run × 30-45 min = 0.5-0.75 hours
- **Total: 4.5-7 hours**

**Phase 2 (Historical Inference):**
- 25 runs × 1-2 min = 25-50 minutes
- **Total: 0.5-1 hour**

**Grand Total: 5-8 hours**

---

## Quick Verification Commands

```bash
# Check if Airflow is running
ps aux | grep airflow

# View DAG status
airflow dags list-runs -d data_pipeline_dag | head -30

# Check specific task logs
airflow tasks logs data_pipeline_dag run_model_training 2025-01-01

# Count successful runs
airflow dags list-runs -d data_pipeline_dag --state success | wc -l

# Web UI
# Open: http://localhost:8080 (login: admin/admin)
```

---

## Summary

✅ **Phase 1**: Trains model on all historical data (2023-01 to 2025-01)
✅ **Phase 2**: Generates predictions for all 25 months using trained model
✅ **Result**: Complete ML pipeline with monitoring capability

This fulfills all three requirements:
- (a) ✅ Train and store model in model bank
- (b) ✅ Retrieve model and make predictions
- (c) ✅ Monitor performance across time (25 months of predictions)
