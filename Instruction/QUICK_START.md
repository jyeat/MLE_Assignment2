# 🚀 QUICK START - Airflow ML Pipeline

## ✅ Pre-Flight Checklist

All systems ready:
- ✅ Predictions directory created: `/app/predictions/`
- ✅ Model bank exists: `/app/models/`
- ✅ Dependencies installed: xgboost, joblib
- ✅ DAG configured: [/app/dags/dag.py](dags/dag.py)
- ✅ Utils ready: model_training.py, model_inference.py

## 🎯 Your Question Answered

**Q: "Since I only have data until 2025-01, does that mean I won't have model inference?"**

**A: You WILL have inference! Here's what happens:**

### On 2025-01-01 execution:
1. ✅ Data pipeline processes 2025-01 data
2. ✅ **Model trains** (on ALL 25 months of data)
3. ✅ **Inference runs** for 2025-01 (model now exists!)

**Result:** You get 1 prediction file for 2025-01

### To get predictions for ALL months (monitoring requirement):
Run **Phase 2** after initial backfill to generate predictions for months 2023-01 through 2025-01 using the trained model.

**Result:** 25 prediction files (complete time series for monitoring)

---

## 📋 Two-Command Setup

### Start Airflow & Run Phase 1:
```bash
# Start services
export AIRFLOW_HOME=/app
airflow webserver --port 8080 -D
airflow scheduler -D
sleep 10

# Run initial backfill (trains model on 2025-01-01)
airflow dags backfill \
    --start-date 2023-01-01 \
    --end-date 2025-01-01 \
    --reset-dagruns \
    data_pipeline_dag
```

### After Phase 1 completes, run Phase 2:
```bash
# Generate predictions for all historical months
airflow tasks clear \
    --yes \
    --only-failed false \
    --task-regex "run_model_inference" \
    --start-date 2023-01-01 \
    --end-date 2025-01-01 \
    data_pipeline_dag
```

---

## 📊 What You'll Get

### After Phase 1:
- `/app/models/` - Model artifacts (trained on 2025-01-01)
- `/app/predictions/predictions_2025-01-01_*.csv` - 1 prediction file

### After Phase 2:
- `/app/predictions/` - 25 prediction files (2023-01 to 2025-01)
- Complete time series for performance monitoring

---

## 🔍 Monitor Progress

```bash
# Web UI
http://localhost:8080 (admin/admin)

# Command line
airflow dags list-runs -d data_pipeline_dag

# Check specific task
airflow tasks logs data_pipeline_dag run_model_training 2025-01-01
```

---

## ⏱️ Estimated Time
- Phase 1: 4.5-7 hours
- Phase 2: 0.5-1 hour
- **Total: 5-8 hours**

---

## 📖 Full Instructions
See [AIRFLOW_INSTRUCTIONS.md](AIRFLOW_INSTRUCTIONS.md) for complete details.

---

## ✅ Requirements Fulfilled

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| (a) Train ML model, store in model bank | Trains on 2025-01-01, stores in `/app/models/` | ✅ |
| (b) Retrieve model, make predictions | Loads from model bank, runs inference | ✅ |
| (c) Monitor performance across time | 25 prediction files for time-series analysis | ✅ |

**YOU'RE READY TO RUN!** 🎉
