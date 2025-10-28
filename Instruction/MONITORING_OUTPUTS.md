# Model Monitoring Outputs

## Overview
The monitoring pipeline automatically generates comprehensive reports with visualizations saved to `/app/monitoring_reports/`

## Generated Files

### 1. CSV Data Files
Located in: `/app/monitoring_reports/`

- **`psi_results.csv`** - Population Stability Index by month
  - Columns: `date`, `psi`, `baseline_date`
  - Shows drift from baseline month

- **`feature_drift_results.csv`** - Feature-level drift metrics
  - Columns: `date`, `feature`, `psi`, `mean_change_pct`, `std_change_pct`
  - Tracks top 10 most important features

- **`performance_metrics.csv`** - Model performance over time (if labels available)
  - Columns: `month`, `n_samples`, `auc`, `accuracy`, `precision`, `recall`, `f1_score`,
    `actual_default_rate`, `predicted_default_rate`, `true_positives`, `false_positives`,
    `true_negatives`, `false_negatives`, `false_negative_rate`, `false_positive_rate`

- **`data_quality_metrics.csv`** - Data quality monitoring
  - Columns: `month`, `n_rows`, `missing_rate_pct`, `n_duplicates`

- **`monitoring_summary.json`** - Complete summary with all metrics, alerts, and recommendations
  - Structured JSON with all monitoring results

### 2. Visualization Files (PNG)
Located in: `/app/monitoring_reports/`

- **`psi_over_time.png`** - PSI trend chart
  - Line plot showing PSI evolution
  - Color-coded regions (stable/moderate/significant drift)
  - Threshold lines at 0.1 and 0.2

- **`performance_dashboard.png`** - 2x2 dashboard with:
  - **Top Left**: AUC trend over time with training baseline
  - **Top Right**: Precision, Recall, F1-Score trends
  - **Bottom Left**: Actual vs Predicted default rates (bar chart)
  - **Bottom Right**: False Positive/Negative rates over time

- **`feature_drift_heatmap.png`** - Feature drift heatmap
  - Rows: Top 10 features
  - Columns: Months
  - Color intensity shows PSI values (green=stable, red=drift)

- **`data_quality_trends.png`** - 1x2 dashboard with:
  - **Left**: Missing value rate over time
  - **Right**: Number of records over time

- **`prediction_statistics.png`** - 1x2 dashboard with:
  - **Left**: Number of customers scored per month
  - **Right**: Predicted default rate over time

### 3. HTML Report
Located in: `/app/monitoring_reports/monitoring_report.html`

**Comprehensive interactive HTML report** containing:

#### Sections:
1. **Executive Summary**
   - Key metric cards (Average PSI, Drift Periods, Average AUC, Total Alerts)
   - Visual dashboard with color-coded metrics

2. **Alerts**
   - Color-coded by severity (Critical/High/Warning)
   - Each alert includes message and recommendation

3. **Population Stability**
   - PSI summary statistics
   - Status indicator (Stable/Drift Detected)
   - Embedded PSI visualization

4. **Model Performance**
   - Performance metrics table (AUC, Accuracy, Precision, Recall, F1)
   - Comparison with training baseline
   - Embedded performance dashboard

5. **Feature Drift**
   - Number of features monitored
   - Features with significant drift
   - Embedded feature drift heatmap

6. **Data Quality**
   - Missing rate statistics
   - Records per month
   - Embedded data quality trends

7. **Prediction Statistics**
   - Embedded prediction statistics charts

8. **Recommendations**
   - Actionable recommendations based on monitoring results
   - Prioritized by severity

#### Features:
- Self-contained (all images embedded as base64)
- Responsive design
- Professional styling
- Can be shared via email or viewed in browser

## Visualization Reference (from model_monitoring.ipynb)

The visualizations are based on your notebook and include:

### PSI Visualization
- X-axis: Month
- Y-axis: PSI value
- Color zones: Green (stable), Orange (moderate), Red (significant drift)
- Reference lines at 0.1 and 0.2 thresholds

### Performance Dashboard
Similar to **Cell #19** in your notebook:
- 4-panel layout
- AUC trend with training baseline comparison
- Multi-metric trends (Precision/Recall/F1)
- Bar charts for default rates
- Error rates over time

### Feature Drift Heatmap
Similar to **Cell #28** in your notebook:
- Heatmap format
- Features as rows, months as columns
- Color scale: Green-Yellow-Red (0.0-0.3 PSI)
- Annotations showing exact PSI values

### Data Quality Trends
Similar to **Cell #31** in your notebook:
- Line plots showing trends
- Missing rate with 5% threshold line
- Row counts to detect data availability issues

## How to Access Reports

### View in Airflow:
1. Navigate to monitoring_pipeline_dag in Airflow UI
2. Check task logs for summary statistics
3. Reports saved to `/app/monitoring_reports/`

### View HTML Report:
```bash
# From container
open /app/monitoring_reports/monitoring_report.html

# Or copy to host and view in browser
docker cp <container>:/app/monitoring_reports/monitoring_report.html .
```

### View Visualizations:
```bash
# List all generated files
ls -lh /app/monitoring_reports/

# View specific visualization
open /app/monitoring_reports/psi_over_time.png
```

## Alert Thresholds

Configurable in `monitoring_pipeline_dag.py`:

```python
config = {
    'psi_threshold': 0.2,              # Alert when PSI >= 0.2
    'auc_drop_threshold': 0.05,        # Alert when AUC drops > 5%
    'missing_rate_threshold': 5.0      # Alert when missing rate > 5%
}
```

## Alerting Logic

### Critical Alerts:
- PSI ≥ 0.2 (significant population drift)
- AUC drop > 5% from training baseline
- N features with PSI ≥ 0.2

### Warning Alerts:
- PSI 0.1-0.2 (moderate drift)
- Missing rate > 5%
- High variability in predictions

## Retraining Recommendations

Automatic recommendation to retrain when:
1. ≥ 3 periods with PSI ≥ 0.2
2. AUC drops > 5% from training
3. Multiple critical alerts detected

## Example Usage

After monitoring DAG completes:

```bash
# View summary
cat /app/monitoring_reports/monitoring_summary.json | jq '.alerts'

# Check PSI results
head /app/monitoring_reports/psi_results.csv

# Open HTML report (if in local environment)
open /app/monitoring_reports/monitoring_report.html
```

## Integration with Jupyter Notebook

You can also load the monitoring results in your notebook:

```python
import pandas as pd
import json

# Load results
psi_df = pd.read_csv('/app/monitoring_reports/psi_results.csv')
perf_df = pd.read_csv('/app/monitoring_reports/performance_metrics.csv')

with open('/app/monitoring_reports/monitoring_summary.json') as f:
    summary = json.load(f)

# View alerts
for alert in summary['alerts']:
    print(f"[{alert['severity']}] {alert['message']}")
```

## Automation

The monitoring pipeline runs automatically:
- **Schedule**: 2nd of each month at midnight
- **Dependency**: Waits for main pipeline to complete
- **Catchup**: Yes (backfills historical monitoring)
- **Notifications**: Email sent on failure

## Troubleshooting

If visualizations are not generated:
1. Check matplotlib is installed
2. Verify sufficient disk space in `/app/monitoring_reports/`
3. Check task logs in Airflow for errors

If HTML report is empty:
1. Verify visualization PNGs were created
2. Check file permissions on `/app/monitoring_reports/`
3. Review logs for image encoding errors
