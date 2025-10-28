# ML Pipeline Monitoring Implementation Summary

## 🎯 Objective
Create an end-to-end ML model monitoring solution that:
1. Monitors model performance and stability
2. Detects population and feature drift
3. Tracks data quality
4. Generates automated alerts and recommendations
5. Produces comprehensive reports with visualizations

## ✅ What Was Implemented

### 1. Model Monitoring Utility ([/app/utils/model_monitoring.py](utils/model_monitoring.py))

**Comprehensive monitoring library** with the following capabilities:

#### PSI (Population Stability Index) Calculation
- `calculate_psi()` - Core PSI calculation
- `calculate_psi_over_time()` - PSI trend analysis
- `get_psi_summary()` - PSI statistics
- **Interpretation**:
  - PSI < 0.1: Stable
  - 0.1 ≤ PSI < 0.2: Moderate drift
  - PSI ≥ 0.2: Significant drift (retrain recommended)

#### Feature Drift Detection
- `calculate_feature_drift()` - Monitors top N features
- `get_feature_drift_summary()` - Summary statistics
- Tracks distribution changes in most important features

#### Model Performance Evaluation
- `evaluate_model_performance()` - Merges predictions with labels
- `get_performance_summary()` - AUC, accuracy, precision, recall, F1
- Compares production performance vs training baseline

#### Data Quality Monitoring
- `monitor_data_quality()` - Missing values, duplicates, row counts
- `get_data_quality_summary()` - Quality metrics summary

#### Alerting System
- `generate_alerts()` - Creates alerts based on thresholds
- `generate_recommendations()` - Actionable recommendations
- Severity levels: critical, high, warning

#### Visualization Generation
- `generate_monitoring_visualizations()` - Creates 5 PNG charts:
  1. PSI over time with color-coded drift zones
  2. Performance dashboard (4-panel: AUC, metrics, default rates, error rates)
  3. Feature drift heatmap
  4. Data quality trends
  5. Prediction statistics

#### HTML Report Generation
- `generate_html_report()` - Interactive HTML report
- Self-contained with base64-encoded images
- Professional styling with metric cards
- Color-coded alerts
- All visualizations embedded

#### Main Workflow
- `run_model_monitoring()` - Orchestrates entire monitoring workflow
- `should_retrain_model()` - Determines if retraining needed

### 2. Monitoring Pipeline DAG ([/app/dags/monitoring_pipeline_dag.py](dags/monitoring_pipeline_dag.py))

**Simplified for school assignment** - No dependencies on main pipeline

#### Key Features:
- **Schedule**: Manual trigger (`schedule_interval=None`)
- **Prerequisites Check**: Verifies model + predictions exist (minimum 2 months)
- **Independent Execution**: Runs on existing predictions in `/app/predictions/`
- **No ExternalTaskSensor**: Simplified to avoid complexity
- **Branching Logic**: Skips if prerequisites not met
- **Retraining Recommendations**: Automatically evaluates if retraining needed

#### Task Flow:
```
start → check_prerequisites → [skip_monitoring, run_monitoring]
                                      ↓
                               check_retraining → [alert_retraining, monitoring_complete]
                                                          ↓
                                                   print_summary → end
```

#### Configuration:
```python
{
    'predictions_dir': '/app/predictions',
    'model_dir': '/app/models',
    'feature_store_path': '/app/datamart/gold/feature_store/',
    'label_store_path': '/app/datamart/gold/label_store/',
    'output_dir': '/app/monitoring_reports',
    'psi_threshold': 0.2,              # Alert if PSI ≥ 0.2
    'auc_drop_threshold': 0.05,        # Alert if AUC drops > 5%
    'missing_rate_threshold': 5.0      # Alert if missing > 5%
}
```

### 3. Main Pipeline Enhancements ([/app/dags/dag.py](dags/dag.py:150-159))

**Fix #1: Error Notifications**
- Increased retries: 1 → 3
- Added `email_on_failure: True`
- Added `email_on_retry: False`
- Added email configuration
- Added `execution_timeout: 4 hours`

### 4. Spark Session Cleanup ([/app/utils/bronze_layer.py](utils/bronze_layer.py:93-127))

**Fix #2: Resource Leak Prevention**
- Added `try/finally` block in `process_bronze_table_main()`
- Ensures `spark.stop()` always executes
- Prevents Spark session accumulation
- Proper error handling

## 📊 Generated Outputs

### Directory: `/app/monitoring_reports/`

#### CSV Files (Machine-Readable):
1. `psi_results.csv` - PSI by month
2. `feature_drift_results.csv` - Feature drift metrics
3. `performance_metrics.csv` - Monthly performance (if labels available)
4. `data_quality_metrics.csv` - Data quality metrics
5. `monitoring_summary.json` - Complete summary

#### Visualizations (PNG):
1. `psi_over_time.png` - PSI trend chart
2. `performance_dashboard.png` - 4-panel dashboard
3. `feature_drift_heatmap.png` - Feature drift heatmap
4. `data_quality_trends.png` - Quality trends
5. `prediction_statistics.png` - Prediction stats

#### Interactive Report (HTML):
- `monitoring_report.html` - **Main deliverable**
  - Self-contained (images embedded)
  - Professional design
  - Executive summary with metric cards
  - Color-coded alerts
  - All visualizations
  - Actionable recommendations
  - Can be shared via email or viewed in browser

## 🚀 How to Use

### Step 1: Run Main Pipeline
Ensure you have predictions:
```bash
# Check if predictions exist
ls /app/predictions/

# Should have multiple predictions_YYYY-MM-DD.csv files
```

### Step 2: Trigger Monitoring DAG
In Airflow UI:
1. Navigate to `monitoring_pipeline_dag`
2. Click "Trigger DAG" (play button)
3. Wait for completion (~5-10 minutes)

Or via CLI:
```bash
airflow dags trigger monitoring_pipeline_dag
```

### Step 3: View Results

**Option 1: HTML Report (Recommended)**
```bash
# Copy to local machine
docker cp <container>:/app/monitoring_reports/monitoring_report.html .

# Open in browser
open monitoring_report.html
```

**Option 2: Airflow Logs**
- View task logs for `print_summary` task
- Shows executive summary and alerts

**Option 3: Direct File Access**
```bash
# View JSON summary
cat /app/monitoring_reports/monitoring_summary.json | jq

# View visualizations
ls /app/monitoring_reports/*.png

# Load in Jupyter
import pandas as pd
psi_df = pd.read_csv('/app/monitoring_reports/psi_results.csv')
```

## 📈 Monitoring Metrics Explained

### Population Stability Index (PSI)
**What**: Measures distribution shift from baseline month
**Why**: Detects if customer population changed
**Action**: If PSI ≥ 0.2 → Retrain model

### Feature Drift
**What**: PSI for individual features
**Why**: Identifies which features changed
**Action**: Investigate root causes in data pipeline

### Model Performance
**What**: AUC, Precision, Recall on production data
**Why**: Validates model still performs well
**Action**: If AUC drops > 5% → Investigate

### Data Quality
**What**: Missing rates, row counts
**Why**: Detects data pipeline issues
**Action**: If missing > 5% → Check upstream systems

## 🔔 Alert Thresholds

### Critical Alerts:
- PSI ≥ 0.2 in any period
- AUC drop > 5% from training
- Multiple features with high drift

### Warning Alerts:
- PSI 0.1-0.2 (moderate drift)
- Missing rate > 5%
- High variability in predictions

## 🔄 Retraining Recommendations

Model retraining automatically recommended when:
1. PSI ≥ 0.2 in 3+ periods
2. AUC drops > 5% from training baseline
3. Multiple critical alerts detected

See `should_retrain_model()` function for logic.

## 📝 Key Design Decisions

### 1. Separate Monitoring DAG (Not Integrated)
**Why**:
- Monitoring failures don't block inference
- Can run independently on accumulated predictions
- Easier to trigger on-demand for testing
- Better separation of concerns

### 2. No ExternalTaskSensor (Simplified)
**Why**:
- Reduces complexity for school assignment
- No dependency on main pipeline timing
- Can test monitoring anytime predictions exist
- More flexible execution

### 3. Manual Trigger (No Schedule)
**Why**:
- Assignment can be tested anytime
- No accidental runs consuming resources
- Professor can trigger on-demand for grading

### 4. HTML Report with Embedded Images
**Why**:
- Single file can be shared via email
- No external dependencies
- Works offline
- Professional presentation

## 🎓 Assignment Context

This implementation demonstrates:
1. **MLOps Best Practices**: Automated monitoring, alerting, reporting
2. **Data Engineering**: PySpark, Parquet, data quality checks
3. **ML Model Lifecycle**: Training, inference, monitoring, retraining triggers
4. **Airflow Orchestration**: DAG design, branching, XCom, task dependencies
5. **Visualization**: matplotlib, seaborn, HTML report generation
6. **Software Engineering**: Modular code, error handling, resource cleanup

## 🐛 Troubleshooting

### Issue: "No predictions found"
**Solution**: Run main pipeline first to generate predictions

### Issue: "Only 1 prediction file"
**Solution**: Need at least 2 months for trend analysis

### Issue: "No labeled data for evaluation"
**Solution**: Normal - only first N months have labels. PSI and data quality monitoring still work.

### Issue: Visualizations not generated
**Solution**: Check `/app/monitoring_reports/` permissions and disk space

### Issue: HTML report empty
**Solution**: Check PNG files were created first

## 📚 Files Modified/Created

### Created:
- ✅ `/app/utils/model_monitoring.py` (1350+ lines)
- ✅ `/app/dags/monitoring_pipeline_dag.py` (simplified version)
- ✅ `/app/MONITORING_OUTPUTS.md` (documentation)
- ✅ `/app/IMPLEMENTATION_SUMMARY.md` (this file)

### Modified:
- ✅ `/app/dags/dag.py` (lines 150-159: error notifications)
- ✅ `/app/utils/bronze_layer.py` (lines 93-127: Spark cleanup)

### Backups Created:
- `/app/utils/model_monitoring_backup.py`
- `/app/dags/monitoring_pipeline_dag_backup.py`

## 🎯 Success Criteria

✅ **Functional Requirements**:
- [x] Model training and storage (main DAG)
- [x] Model inference generation (main DAG)
- [x] Model performance monitoring (monitoring DAG)
- [x] Drift detection (monitoring DAG)
- [x] Automated alerting (monitoring DAG)
- [x] Comprehensive reporting (monitoring DAG)

✅ **Technical Requirements**:
- [x] Error handling and notifications
- [x] Resource cleanup (Spark sessions)
- [x] Modular, reusable code
- [x] Professional visualizations
- [x] Clear documentation

✅ **Assignment Requirements**:
- [x] End-to-end ML pipeline
- [x] Airflow orchestration
- [x] Data quality monitoring
- [x] Model stability tracking
- [x] Retraining recommendations

## 🎉 Summary

Your ML pipeline now has **production-grade monitoring** with:
- Automated drift detection
- Performance tracking
- Data quality monitoring
- Visual reports with HTML dashboard
- Smart alerting and recommendations
- Simplified execution for assignment demo

**Next Step**: Trigger `monitoring_pipeline_dag` in Airflow and view the generated `monitoring_report.html`! 📊
