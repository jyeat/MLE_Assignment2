"""
Visualization Utilities for Model Monitoring

Generates visualizations and HTML reports from gold table monitoring data.
Separated from monitoring calculations for better separation of concerns.

Author: ML Pipeline Team
Date: 2025-11-02
"""

import os
import glob
import json
import base64
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def generate_monitoring_visualizations_from_gold(
    gold_monitoring_store: str,
    output_dir: str,
    timestamp: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate comprehensive monitoring visualizations from gold table data

    Args:
        gold_monitoring_store: Path to gold/monitoring_store directory
        output_dir: Directory to save visualization images (PNG files)
        timestamp: Optional specific timestamp to use, otherwise uses latest

    Returns:
        Dictionary with paths to generated visualization files
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import seaborn as sns

    from pyspark.sql import SparkSession

    # Initialize Spark session for reading Parquet files
    spark = SparkSession.builder \
        .appName("MonitoringVisualization") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    try:
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100

        os.makedirs(output_dir, exist_ok=True)
        visualization_paths = {}

        # Find latest files if timestamp not specified
        if timestamp is None:
            # Get latest timestamp from files
            psi_files = glob.glob(f"{gold_monitoring_store}/psi_results_*.parquet")
            if psi_files:
                latest_psi = max(psi_files, key=os.path.getmtime)
                timestamp = os.path.basename(latest_psi).replace('psi_results_', '').replace('.parquet', '')

        print(f"📊 Generating visualizations from gold table (timestamp: {timestamp})")

        # Load data from gold table
        psi_path = f"{gold_monitoring_store}/psi_results_{timestamp}.parquet"
        performance_path = f"{gold_monitoring_store}/performance_metrics_{timestamp}.parquet"
        drift_path = f"{gold_monitoring_store}/feature_drift_{timestamp}.parquet"
        quality_path = f"{gold_monitoring_store}/data_quality_{timestamp}.parquet"
        summary_path = f"{gold_monitoring_store}/monitoring_summary_{timestamp}.json"

        # Load PSI data
        psi_df = None
        if os.path.exists(psi_path):
            psi_spark_df = spark.read.parquet(psi_path)
            psi_df = psi_spark_df.toPandas()
            print(f"   ✅ Loaded PSI data: {len(psi_df)} records")

        # Load performance data
        performance_df = None
        if os.path.exists(performance_path):
            perf_spark_df = spark.read.parquet(performance_path)
            performance_df = perf_spark_df.toPandas()
            print(f"   ✅ Loaded performance data: {len(performance_df)} records")

        # Load feature drift data
        feature_drift_df = None
        if os.path.exists(drift_path):
            drift_spark_df = spark.read.parquet(drift_path)
            feature_drift_df = drift_spark_df.toPandas()
            print(f"   ✅ Loaded feature drift data: {len(feature_drift_df)} records")

        # Load data quality data
        data_quality_df = None
        if os.path.exists(quality_path):
            quality_spark_df = spark.read.parquet(quality_path)
            data_quality_df = quality_spark_df.toPandas()
            print(f"   ✅ Loaded data quality data: {len(data_quality_df)} records")

        # Load summary for metadata
        metadata = None
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
                # Extract model metadata if available
                metadata = summary_data.get('metadata', {})

        # Load all predictions from gold prediction store for distribution analysis
        gold_prediction_store = gold_monitoring_store.replace('monitoring_store', 'prediction_store')
        all_predictions = None
        if os.path.exists(gold_prediction_store):
            pred_files = glob.glob(f"{gold_prediction_store}/gold_prediction_store_*.parquet")
            if pred_files:
                all_pred_dfs = []
                for pred_file in pred_files:
                    pred_spark_df = spark.read.parquet(pred_file)
                    all_pred_dfs.append(pred_spark_df.toPandas())
                if all_pred_dfs:
                    all_predictions = pd.concat(all_pred_dfs, ignore_index=True)
                    print(f"   ✅ Loaded predictions: {len(all_predictions)} records from {len(pred_files)} files")

        # 1. PSI Over Time
        if psi_df is not None and len(psi_df) > 0:
            # Sort by date to ensure chronological order
            psi_df = psi_df.sort_values('date').reset_index(drop=True)

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(psi_df['date'], psi_df['psi'], marker='o', linewidth=2.5, markersize=8, color='#2E86AB')

            # Add threshold lines
            ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=2, label='Moderate Drift (0.1)')
            ax.axhline(y=0.2, color='red', linestyle='--', linewidth=2, label='Significant Drift (0.2)')

            # Color regions
            ax.fill_between(psi_df['date'], 0, 0.1, alpha=0.2, color='green', label='Stable')
            ax.fill_between(psi_df['date'], 0.1, 0.2, alpha=0.2, color='orange')
            if psi_df['psi'].max() > 0.2:
                ax.fill_between(psi_df['date'], 0.2, psi_df['psi'].max(), alpha=0.2, color='red')

            ax.set_title('Population Stability Index (PSI) Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('PSI', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            psi_path = f"{output_dir}/psi_over_time.png"
            plt.savefig(psi_path, bbox_inches='tight')
            plt.close()
            visualization_paths['psi'] = psi_path
            print(f"   📈 PSI chart: {os.path.basename(psi_path)}")

        # 2. Model Performance Dashboard (if available)
        if performance_df is not None and len(performance_df) > 0:
            # Sort by month to ensure chronological order
            performance_df = performance_df.sort_values('month').reset_index(drop=True)

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # AUC Trend
            axes[0, 0].plot(performance_df['month'], performance_df['auc'],
                           marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label='Monthly AUC')
            if metadata and 'metrics' in metadata and 'test' in metadata['metrics']:
                test_auc = metadata['metrics']['test']['auc']
                axes[0, 0].axhline(y=test_auc, color='green', linestyle='--', linewidth=2,
                                  label=f'Training Test AUC ({test_auc:.3f})')
            axes[0, 0].set_title('AUC-ROC Over Time', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('AUC')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylim([0.6, 1.0])

            # Precision, Recall, F1
            axes[0, 1].plot(performance_df['month'], performance_df['precision'],
                           marker='o', linewidth=2, label='Precision', color='#A23B72')
            axes[0, 1].plot(performance_df['month'], performance_df['recall'],
                           marker='s', linewidth=2, label='Recall', color='#F18F01')
            axes[0, 1].plot(performance_df['month'], performance_df['f1_score'],
                           marker='^', linewidth=2, label='F1-Score', color='#C73E1D')
            axes[0, 1].set_title('Classification Metrics', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_ylim([0.5, 1.0])

            # Default Rate Comparison
            x = np.arange(len(performance_df))
            width = 0.35
            axes[1, 0].bar(x - width/2, performance_df['actual_default_rate']*100, width,
                          label='Actual', color='#E63946', alpha=0.8)
            axes[1, 0].bar(x + width/2, performance_df['predicted_default_rate']*100, width,
                          label='Predicted', color='#457B9D', alpha=0.8)
            axes[1, 0].set_title('Actual vs Predicted Default Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Default Rate (%)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([m.strftime('%Y-%m') for m in performance_df['month']],
                                       rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # False Positive/Negative Rates
            axes[1, 1].plot(performance_df['month'], performance_df['false_positive_rate']*100,
                           marker='o', linewidth=2, label='False Positive Rate', color='#F4A261')
            axes[1, 1].plot(performance_df['month'], performance_df['false_negative_rate']*100,
                           marker='s', linewidth=2, label='False Negative Rate', color='#E76F51')
            axes[1, 1].set_title('Error Rates Over Time', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Error Rate (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()

            perf_path = f"{output_dir}/performance_dashboard.png"
            plt.savefig(perf_path, bbox_inches='tight')
            plt.close()
            visualization_paths['performance'] = perf_path
            print(f"   📈 Performance dashboard: {os.path.basename(perf_path)}")

        # 3. Feature Drift Heatmap
        if feature_drift_df is not None and len(feature_drift_df) > 0:
            psi_pivot = feature_drift_df.pivot(index='feature', columns='date', values='psi')

            fig, ax = plt.subplots(figsize=(16, 8))
            sns.heatmap(psi_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                       center=0.1, vmin=0, vmax=0.3, ax=ax, cbar_kws={'label': 'PSI'})
            ax.set_title('Feature Drift (PSI) Heatmap', fontsize=14, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            plt.tight_layout()

            drift_path = f"{output_dir}/feature_drift_heatmap.png"
            plt.savefig(drift_path, bbox_inches='tight')
            plt.close()
            visualization_paths['feature_drift'] = drift_path
            print(f"   📈 Feature drift heatmap: {os.path.basename(drift_path)}")

        # 4. Data Quality Trends
        if data_quality_df is not None and len(data_quality_df) > 0:
            # Sort by month to ensure chronological order
            data_quality_df = data_quality_df.sort_values('month').reset_index(drop=True)

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            # Missing rate
            axes[0].plot(data_quality_df['month'], data_quality_df['missing_rate_pct'],
                        marker='o', linewidth=2.5, markersize=7, color='#E63946')
            axes[0].axhline(y=5, color='red', linestyle='--', linewidth=2, label='Alert Threshold (5%)')
            axes[0].set_title('Missing Value Rate Over Time', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Month')
            axes[0].set_ylabel('Missing Rate (%)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)

            # Row count
            axes[1].plot(data_quality_df['month'], data_quality_df['n_rows'],
                        marker='o', linewidth=2.5, markersize=7, color='#457B9D')
            axes[1].set_title('Number of Records Over Time', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Month')
            axes[1].set_ylabel('Number of Records')
            axes[1].grid(True, alpha=0.3)
            axes[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()

            quality_path = f"{output_dir}/data_quality_trends.png"
            plt.savefig(quality_path, bbox_inches='tight')
            plt.close()
            visualization_paths['data_quality'] = quality_path
            print(f"   📈 Data quality trends: {os.path.basename(quality_path)}")

        # 5. Prediction Distribution
        if all_predictions is not None and len(all_predictions) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Monthly statistics
            monthly_stats = all_predictions.groupby('inference_date').agg({
                'Customer_ID': 'count',
                'default_probability': 'mean',
                'predicted_default': 'mean'
            }).reset_index()
            monthly_stats.columns = ['date', 'n_customers', 'avg_prob', 'default_rate']

            # Sort by date to ensure chronological order
            monthly_stats = monthly_stats.sort_values('date').reset_index(drop=True)

            # Customers scored
            axes[0].plot(monthly_stats['date'], monthly_stats['n_customers'],
                        marker='o', linewidth=2.5, markersize=7, color='#2E86AB')
            axes[0].set_title('Customers Scored per Month', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Month')
            axes[0].set_ylabel('Number of Customers')
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)

            # Average probability and default rate
            ax2 = axes[1]
            ax2.plot(monthly_stats['date'], monthly_stats['avg_prob']*100,
                    marker='o', linewidth=2.5, markersize=7, color='#E76F51', label='Avg Probability')
            ax2.plot(monthly_stats['date'], monthly_stats['default_rate']*100,
                    marker='s', linewidth=2.5, markersize=7, color='#E63946', label='Predicted Default Rate')
            ax2.set_title('Default Prediction Trends', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Rate (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            pred_path = f"{output_dir}/prediction_distribution.png"
            plt.savefig(pred_path, bbox_inches='tight')
            plt.close()
            visualization_paths['prediction_dist'] = pred_path
            print(f"   📈 Prediction distribution: {os.path.basename(pred_path)}")

        print(f"\n✅ Generated {len(visualization_paths)} visualizations in {output_dir}")

        return visualization_paths

    finally:
        spark.stop()


def generate_html_report_from_gold(
    gold_monitoring_store: str,
    visualization_paths: Dict[str, str],
    output_dir: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate comprehensive HTML report from gold table data with visualizations

    Args:
        gold_monitoring_store: Path to gold/monitoring_store directory
        visualization_paths: Paths to visualization images
        output_dir: Directory to save HTML report
        timestamp: Optional specific timestamp to use, otherwise uses latest

    Returns:
        Path to generated HTML report
    """
    # Find latest summary file if timestamp not specified
    if timestamp is None:
        summary_files = glob.glob(f"{gold_monitoring_store}/monitoring_summary_*.json")
        if summary_files:
            latest_summary = max(summary_files, key=os.path.getmtime)
            timestamp = os.path.basename(latest_summary).replace('monitoring_summary_', '').replace('.json', '')

    summary_path = f"{gold_monitoring_store}/monitoring_summary_{timestamp}.json"

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary report not found: {summary_path}")

    # Load summary report
    with open(summary_path, 'r') as f:
        summary_report = json.load(f)

    print(f"📄 Generating HTML report from gold table (timestamp: {timestamp})")

    # Helper to encode images as base64
    def img_to_base64(img_path):
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        return ""

    # Extract summaries
    psi_summary = summary_report.get('psi_summary', {})
    performance_summary = summary_report.get('performance_summary', {})
    feature_drift_summary = summary_report.get('feature_drift_summary', {})
    data_quality_summary = summary_report.get('data_quality_summary', {})
    alerts = summary_report.get('alerts', [])
    recommendations = summary_report.get('recommendations', [])
    report_timestamp = summary_report.get('timestamp', datetime.now().isoformat())

    # Build HTML content
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Model Monitoring Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .summary-card {{
            background-color: #ecf0f1;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .alert {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-critical {{
            background-color: #fadbd8;
            border-left-color: #e74c3c;
        }}
        .alert-warning {{
            background-color: #fcf3cf;
            border-left-color: #f39c12;
        }}
        .alert-info {{
            background-color: #d6eaf8;
            border-left-color: #3498db;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .visualization {{
            margin: 20px 0;
            text-align: center;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .recommendation {{
            background-color: #e8f8f5;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #1abc9c;
            border-radius: 5px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Model Monitoring Report</h1>
        <p><strong>Generated:</strong> {report_timestamp}</p>
        <p><strong>Data Source:</strong> Gold Table ({timestamp})</p>

        <h2>📊 Executive Summary</h2>
        <div class="summary-card">
            <div class="metric">
                <div class="metric-label">Average PSI</div>
                <div class="metric-value">{psi_summary.get('avg_psi', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Significant Drift Periods</div>
                <div class="metric-value">{psi_summary.get('n_significant', 0)}</div>
            </div>
"""

    # Add performance metrics if available
    if performance_summary.get('evaluation_available', False):
        avg_metrics = performance_summary.get('avg_metrics', {})
        html_content += f"""
            <div class="metric">
                <div class="metric-label">Average AUC</div>
                <div class="metric-value">{avg_metrics.get('auc', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Average Precision</div>
                <div class="metric-value">{avg_metrics.get('precision', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Average Recall</div>
                <div class="metric-value">{avg_metrics.get('recall', 0):.4f}</div>
            </div>
"""

    html_content += """
        </div>

        <h2>🚨 Alerts</h2>
"""

    if alerts:
        for alert in alerts:
            severity = alert.get('severity', 'info')
            alert_class = f"alert-{severity}"
            html_content += f"""
        <div class="alert {alert_class}">
            <strong>[{severity.upper()}]</strong> {alert.get('message', '')}
        </div>
"""
    else:
        html_content += """
        <p>No alerts generated. All metrics within acceptable ranges.</p>
"""

    html_content += """
        <h2>💡 Recommendations</h2>
"""

    if recommendations:
        for rec in recommendations:
            html_content += f"""
        <div class="recommendation">
            ✓ {rec}
        </div>
"""
    else:
        html_content += """
        <p>No specific recommendations at this time. Continue monitoring.</p>
"""

    # Add visualizations
    html_content += """
        <h2>📈 Visualizations</h2>
"""

    # PSI visualization
    if 'psi' in visualization_paths:
        psi_img = img_to_base64(visualization_paths['psi'])
        html_content += f"""
        <div class="visualization">
            <h3>Population Stability Index (PSI) Over Time</h3>
            <img src="data:image/png;base64,{psi_img}" alt="PSI Over Time">
        </div>
"""

    # Performance dashboard
    if 'performance' in visualization_paths:
        perf_img = img_to_base64(visualization_paths['performance'])
        html_content += f"""
        <div class="visualization">
            <h3>Model Performance Dashboard</h3>
            <img src="data:image/png;base64,{perf_img}" alt="Performance Dashboard">
        </div>
"""

    # Feature drift heatmap
    if 'feature_drift' in visualization_paths:
        drift_img = img_to_base64(visualization_paths['feature_drift'])
        html_content += f"""
        <div class="visualization">
            <h3>Feature Drift Heatmap</h3>
            <img src="data:image/png;base64,{drift_img}" alt="Feature Drift Heatmap">
        </div>
"""

    # Data quality trends
    if 'data_quality' in visualization_paths:
        quality_img = img_to_base64(visualization_paths['data_quality'])
        html_content += f"""
        <div class="visualization">
            <h3>Data Quality Trends</h3>
            <img src="data:image/png;base64,{quality_img}" alt="Data Quality Trends">
        </div>
"""

    # Prediction distribution
    if 'prediction_dist' in visualization_paths:
        pred_img = img_to_base64(visualization_paths['prediction_dist'])
        html_content += f"""
        <div class="visualization">
            <h3>Prediction Distribution</h3>
            <img src="data:image/png;base64,{pred_img}" alt="Prediction Distribution">
        </div>
"""

    # Add detailed metrics tables
    html_content += f"""
        <h2>📋 Detailed Metrics</h2>

        <h3>PSI Analysis</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average PSI</td>
                <td>{psi_summary.get('avg_psi', 0):.4f}</td>
            </tr>
            <tr>
                <td>Maximum PSI</td>
                <td>{psi_summary.get('max_psi', 0):.4f}</td>
            </tr>
            <tr>
                <td>Periods with Significant Drift</td>
                <td>{psi_summary.get('n_significant', 0)}</td>
            </tr>
            <tr>
                <td>Periods with Moderate Drift</td>
                <td>{psi_summary.get('n_moderate', 0)}</td>
            </tr>
        </table>
"""

    if performance_summary.get('evaluation_available', False):
        avg_metrics = performance_summary.get('avg_metrics', {})
        html_content += f"""
        <h3>Performance Metrics</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Average Value</th>
            </tr>
            <tr>
                <td>AUC-ROC</td>
                <td>{avg_metrics.get('auc', 0):.4f}</td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{avg_metrics.get('precision', 0):.4f}</td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{avg_metrics.get('recall', 0):.4f}</td>
            </tr>
            <tr>
                <td>F1-Score</td>
                <td>{avg_metrics.get('f1_score', 0):.4f}</td>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{avg_metrics.get('accuracy', 0):.4f}</td>
            </tr>
        </table>
"""

    html_content += f"""
        <h3>Data Quality</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Average Missing Rate</td>
                <td>{data_quality_summary.get('avg_missing_rate_pct', 0):.2f}%</td>
            </tr>
            <tr>
                <td>Periods with High Missing Rate</td>
                <td>{data_quality_summary.get('n_high_missing', 0)}</td>
            </tr>
        </table>

        <div class="footer">
            <p>This report was automatically generated from the gold monitoring table.</p>
            <p>Data source: {gold_monitoring_store}</p>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML report
    html_path = f"{output_dir}/monitoring_report.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ HTML report: {os.path.basename(html_path)}")

    return html_path
