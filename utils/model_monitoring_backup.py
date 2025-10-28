"""
Model Monitoring Utilities

Comprehensive utilities for monitoring ML model performance, stability, and data quality.
Includes PSI calculation, feature drift detection, performance evaluation, and alerting.

Author: ML Pipeline Team
Date: 2025-10-28
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
import joblib


# ============================================================================
# 1. PSI (POPULATION STABILITY INDEX) CALCULATION
# ============================================================================

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Population Stability Index (PSI)

    PSI Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change, investigate
    - PSI >= 0.2: Significant change, model may need retraining

    Args:
        expected: Baseline distribution (typically training data or first month)
        actual: Current distribution to compare against baseline
        bins: Number of bins for discretization (default: 10)

    Returns:
        Tuple containing:
        - psi: The PSI value
        - breakpoints: Bin edges used for discretization
        - expected_percents: Expected distribution percentages
        - actual_percents: Actual distribution percentages
    """
    # Create bins based on expected distribution
    breakpoints = np.linspace(0, 1, bins + 1)

    # Calculate distributions
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)

    return psi, breakpoints, expected_percents, actual_percents


def calculate_psi_over_time(
    predictions_df: pd.DataFrame,
    score_column: str = 'default_probability',
    date_column: str = 'inference_date',
    bins: int = 10
) -> pd.DataFrame:
    """
    Calculate PSI for each time period against baseline (first period)

    Args:
        predictions_df: DataFrame containing predictions with dates
        score_column: Column name for model scores
        date_column: Column name for dates
        bins: Number of bins for PSI calculation

    Returns:
        DataFrame with columns: [date, psi, baseline_date]
    """
    # Ensure date column is datetime
    predictions_df = predictions_df.copy()
    predictions_df[date_column] = pd.to_datetime(predictions_df[date_column])

    # Get baseline (first month)
    sorted_dates = sorted(predictions_df[date_column].unique())
    baseline_date = sorted_dates[0]
    baseline_scores = predictions_df[predictions_df[date_column] == baseline_date][score_column].values

    # Calculate PSI for each subsequent month
    psi_results = []

    for date in sorted_dates[1:]:
        month_scores = predictions_df[predictions_df[date_column] == date][score_column].values

        if len(month_scores) > 0:
            psi, _, _, _ = calculate_psi(baseline_scores, month_scores, bins=bins)

            psi_results.append({
                'date': date,
                'psi': psi,
                'baseline_date': baseline_date
            })

    return pd.DataFrame(psi_results)


def get_psi_summary(psi_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics from PSI results

    Args:
        psi_df: DataFrame with PSI results from calculate_psi_over_time

    Returns:
        Dictionary with summary statistics
    """
    if len(psi_df) == 0:
        return {
            'avg_psi': 0,
            'max_psi': 0,
            'min_psi': 0,
            'n_stable': 0,
            'n_moderate': 0,
            'n_significant': 0
        }

    return {
        'avg_psi': float(psi_df['psi'].mean()),
        'max_psi': float(psi_df['psi'].max()),
        'min_psi': float(psi_df['psi'].min()),
        'max_psi_date': psi_df.loc[psi_df['psi'].idxmax(), 'date'],
        'min_psi_date': psi_df.loc[psi_df['psi'].idxmin(), 'date'],
        'n_stable': int((psi_df['psi'] < 0.1).sum()),
        'n_moderate': int(((psi_df['psi'] >= 0.1) & (psi_df['psi'] < 0.2)).sum()),
        'n_significant': int((psi_df['psi'] >= 0.2).sum())
    }


# ============================================================================
# 2. FEATURE DRIFT DETECTION
# ============================================================================

def calculate_feature_drift(
    spark: SparkSession,
    feature_store_path: str,
    feature_importance_df: pd.DataFrame,
    top_n_features: int = 10
) -> pd.DataFrame:
    """
    Calculate feature drift (PSI) for top N important features

    Args:
        spark: Active Spark session
        feature_store_path: Path to feature store parquet files
        feature_importance_df: DataFrame with columns ['feature', 'importance']
        top_n_features: Number of top features to monitor

    Returns:
        DataFrame with columns: [date, feature, psi, mean_change_pct, std_change_pct]
    """
    # Load feature store
    df_features = spark.read.parquet(f'{feature_store_path}*.parquet').toPandas()
    df_features['snapshot_date'] = pd.to_datetime(df_features['snapshot_date'])

    # Get top features
    top_features = feature_importance_df.head(top_n_features)['feature'].tolist()

    # Select only numeric features
    numeric_features = [f for f in top_features if f in df_features.select_dtypes(include=[np.number]).columns]

    # Calculate baseline (first month)
    baseline_date = df_features['snapshot_date'].min()
    baseline_data = df_features[df_features['snapshot_date'] == baseline_date]

    feature_drift_results = []

    for date in sorted(df_features['snapshot_date'].unique())[1:]:
        month_data = df_features[df_features['snapshot_date'] == date]

        for feature in numeric_features:
            if feature in month_data.columns and feature in baseline_data.columns:
                # Get values
                baseline_values = baseline_data[feature].dropna().values
                month_values = month_data[feature].dropna().values

                if len(baseline_values) > 0 and len(month_values) > 0:
                    # Calculate PSI
                    psi, _, _, _ = calculate_psi(baseline_values, month_values)

                    # Calculate mean change
                    mean_baseline = baseline_values.mean()
                    mean_month = month_values.mean()
                    mean_change_pct = ((mean_month - mean_baseline) / (abs(mean_baseline) + 1e-10)) * 100

                    # Calculate std change
                    std_baseline = baseline_values.std()
                    std_month = month_values.std()
                    std_change_pct = ((std_month - std_baseline) / (abs(std_baseline) + 1e-10)) * 100

                    feature_drift_results.append({
                        'date': date,
                        'feature': feature,
                        'psi': psi,
                        'mean_change_pct': mean_change_pct,
                        'std_change_pct': std_change_pct
                    })

    return pd.DataFrame(feature_drift_results)


def get_feature_drift_summary(feature_drift_df: pd.DataFrame) -> Dict:
    """
    Summarize feature drift results

    Args:
        feature_drift_df: Output from calculate_feature_drift

    Returns:
        Dictionary with summary statistics
    """
    if len(feature_drift_df) == 0:
        return {
            'n_features_monitored': 0,
            'avg_feature_psi': 0,
            'features_high_drift': []
        }

    avg_psi_by_feature = feature_drift_df.groupby('feature')['psi'].mean().sort_values(ascending=False)
    high_drift_features = feature_drift_df[feature_drift_df['psi'] >= 0.2]['feature'].unique().tolist()

    return {
        'n_features_monitored': len(feature_drift_df['feature'].unique()),
        'avg_feature_psi': float(feature_drift_df['psi'].mean()),
        'top_drifting_features': avg_psi_by_feature.head(5).to_dict(),
        'features_high_drift': high_drift_features,
        'n_features_high_drift': len(high_drift_features)
    }


# ============================================================================
# 3. MODEL PERFORMANCE EVALUATION
# ============================================================================

def evaluate_model_performance(
    spark: SparkSession,
    predictions_df: pd.DataFrame,
    label_store_path: str
) -> pd.DataFrame:
    """
    Evaluate model performance by merging predictions with actual labels

    Args:
        spark: Active Spark session
        predictions_df: DataFrame with predictions
        label_store_path: Path to label store parquet files

    Returns:
        DataFrame with monthly performance metrics
    """
    try:
        # Load labels
        df_labels_spark = spark.read.parquet(f'{label_store_path}*.parquet')
        df_labels = df_labels_spark.toPandas()
        df_labels['snapshot_date'] = pd.to_datetime(df_labels['snapshot_date'])

        # Prepare for merge
        df_labels_for_merge = df_labels[['Customer_ID', 'label', 'label_def', 'snapshot_date']].copy()
        df_labels_for_merge = df_labels_for_merge.rename(columns={'snapshot_date': 'label_date'})

        # Merge predictions with labels
        df_eval = predictions_df.merge(df_labels_for_merge, on='Customer_ID', how='inner')

        # Calculate time difference in months
        df_eval['date_diff_months'] = (
            (df_eval['label_date'].dt.year - df_eval['inference_date'].dt.year) * 12 +
            (df_eval['label_date'].dt.month - df_eval['inference_date'].dt.month)
        )

        # Filter to reasonable time window (-6 to +12 months)
        df_eval = df_eval[(df_eval['date_diff_months'] >= -6) & (df_eval['date_diff_months'] <= 12)]

        # Keep closest match for each customer-inference_date pair
        df_eval['abs_date_diff'] = df_eval['date_diff_months'].abs()
        df_eval = df_eval.sort_values('abs_date_diff').groupby(['Customer_ID', 'inference_date']).first().reset_index()

        # Calculate metrics by month
        monthly_metrics = []

        for month in sorted(df_eval['inference_date'].unique()):
            month_data = df_eval[df_eval['inference_date'] == month]

            # Need at least 2 samples and both classes present
            if len(month_data) >= 2 and month_data['label'].nunique() > 1:
                y_true = month_data['label']
                y_pred = month_data['predicted_default']
                y_prob = month_data['default_probability']

                # Calculate metrics
                auc = roc_auc_score(y_true, y_prob)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                # Business metrics
                actual_default_rate = y_true.mean()
                predicted_default_rate = y_pred.mean()
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                monthly_metrics.append({
                    'month': month,
                    'n_samples': len(month_data),
                    'auc': auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'actual_default_rate': actual_default_rate,
                    'predicted_default_rate': predicted_default_rate,
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'false_negative_rate': fnr,
                    'false_positive_rate': fpr
                })

        return pd.DataFrame(monthly_metrics)

    except Exception as e:
        print(f"Warning: Could not evaluate performance: {e}")
        return pd.DataFrame()


def get_performance_summary(metrics_df: pd.DataFrame, training_metadata: Optional[Dict] = None) -> Dict:
    """
    Summarize model performance metrics

    Args:
        metrics_df: Output from evaluate_model_performance
        training_metadata: Optional metadata from model training

    Returns:
        Dictionary with performance summary
    """
    if len(metrics_df) == 0:
        return {
            'evaluation_available': False,
            'message': 'No labeled data available for evaluation'
        }

    summary = {
        'evaluation_available': True,
        'evaluation_period': {
            'start': metrics_df['month'].min(),
            'end': metrics_df['month'].max()
        },
        'total_samples_evaluated': int(metrics_df['n_samples'].sum()),
        'avg_metrics': {
            'auc': float(metrics_df['auc'].mean()),
            'accuracy': float(metrics_df['accuracy'].mean()),
            'precision': float(metrics_df['precision'].mean()),
            'recall': float(metrics_df['recall'].mean()),
            'f1_score': float(metrics_df['f1_score'].mean())
        },
        'metric_ranges': {
            'auc_min': float(metrics_df['auc'].min()),
            'auc_max': float(metrics_df['auc'].max()),
        },
        'business_metrics': {
            'avg_actual_default_rate': float(metrics_df['actual_default_rate'].mean()),
            'avg_predicted_default_rate': float(metrics_df['predicted_default_rate'].mean()),
            'avg_false_negative_rate': float(metrics_df['false_negative_rate'].mean()),
            'avg_false_positive_rate': float(metrics_df['false_positive_rate'].mean())
        }
    }

    # Compare with training if metadata available
    if training_metadata and 'metrics' in training_metadata:
        if 'test' in training_metadata['metrics'] and 'auc' in training_metadata['metrics']['test']:
            train_auc = training_metadata['metrics']['test']['auc']
            prod_auc = summary['avg_metrics']['auc']
            summary['training_comparison'] = {
                'training_test_auc': train_auc,
                'production_avg_auc': prod_auc,
                'auc_difference': prod_auc - train_auc,
                'performance_status': 'better' if prod_auc > train_auc + 0.01 else 'similar' if abs(prod_auc - train_auc) <= 0.01 else 'degraded'
            }

    return summary


# ============================================================================
# 4. DATA QUALITY MONITORING
# ============================================================================

def monitor_data_quality(spark: SparkSession, feature_store_path: str) -> pd.DataFrame:
    """
    Monitor data quality metrics over time

    Args:
        spark: Active Spark session
        feature_store_path: Path to feature store parquet files

    Returns:
        DataFrame with data quality metrics by month
    """
    # Load feature store
    df_features = spark.read.parquet(f'{feature_store_path}*.parquet').toPandas()
    df_features['snapshot_date'] = pd.to_datetime(df_features['snapshot_date'])

    data_quality_results = []

    for month in sorted(df_features['snapshot_date'].unique()):
        month_data = df_features[df_features['snapshot_date'] == month]

        # Calculate quality metrics
        n_rows = len(month_data)
        n_features = len(month_data.columns) - 2  # Exclude Customer_ID and snapshot_date

        # Missing value rate
        missing_rate = month_data.isnull().sum().sum() / (n_rows * n_features) * 100

        # Number of duplicates
        n_duplicates = month_data.duplicated(subset=['Customer_ID']).sum()

        data_quality_results.append({
            'month': month,
            'n_rows': n_rows,
            'missing_rate_pct': missing_rate,
            'n_duplicates': n_duplicates
        })

    return pd.DataFrame(data_quality_results)


def get_data_quality_summary(data_quality_df: pd.DataFrame) -> Dict:
    """
    Summarize data quality metrics

    Args:
        data_quality_df: Output from monitor_data_quality

    Returns:
        Dictionary with data quality summary
    """
    if len(data_quality_df) == 0:
        return {
            'avg_missing_rate': 0,
            'max_missing_rate': 0,
            'avg_rows_per_month': 0
        }

    return {
        'avg_missing_rate_pct': float(data_quality_df['missing_rate_pct'].mean()),
        'max_missing_rate_pct': float(data_quality_df['missing_rate_pct'].max()),
        'max_missing_rate_date': data_quality_df.loc[data_quality_df['missing_rate_pct'].idxmax(), 'month'],
        'avg_rows_per_month': float(data_quality_df['n_rows'].mean()),
        'total_duplicates': int(data_quality_df['n_duplicates'].sum())
    }


# ============================================================================
# 5. ALERTING AND RECOMMENDATIONS
# ============================================================================

def generate_alerts(
    psi_summary: Dict,
    performance_summary: Dict,
    feature_drift_summary: Dict,
    data_quality_summary: Dict,
    psi_threshold: float = 0.2,
    auc_drop_threshold: float = 0.05,
    missing_rate_threshold: float = 5.0
) -> List[Dict]:
    """
    Generate alerts based on monitoring results

    Args:
        psi_summary: Output from get_psi_summary
        performance_summary: Output from get_performance_summary
        feature_drift_summary: Output from get_feature_drift_summary
        data_quality_summary: Output from get_data_quality_summary
        psi_threshold: PSI threshold for alerts
        auc_drop_threshold: AUC drop threshold
        missing_rate_threshold: Missing rate percentage threshold

    Returns:
        List of alert dictionaries
    """
    alerts = []

    # Check PSI
    if psi_summary.get('n_significant', 0) > 0:
        alerts.append({
            'severity': 'critical',
            'category': 'population_drift',
            'message': f"Significant population drift detected in {psi_summary['n_significant']} periods (PSI >= {psi_threshold})",
            'recommendation': 'Model retraining strongly recommended',
            'metrics': {
                'max_psi': psi_summary['max_psi'],
                'avg_psi': psi_summary['avg_psi']
            }
        })
    elif psi_summary.get('n_moderate', 0) > 0:
        alerts.append({
            'severity': 'warning',
            'category': 'population_drift',
            'message': f"Moderate population drift detected in {psi_summary['n_moderate']} periods",
            'recommendation': 'Monitor closely and investigate root causes'
        })

    # Check model performance
    if performance_summary.get('evaluation_available'):
        if 'training_comparison' in performance_summary:
            comp = performance_summary['training_comparison']
            if comp['performance_status'] == 'degraded' and comp['auc_difference'] < -auc_drop_threshold:
                alerts.append({
                    'severity': 'critical',
                    'category': 'performance_degradation',
                    'message': f"Production AUC ({comp['production_avg_auc']:.4f}) significantly lower than training ({comp['training_test_auc']:.4f})",
                    'recommendation': 'Investigate model degradation and consider retraining',
                    'metrics': {
                        'auc_drop': comp['auc_difference']
                    }
                })

    # Check feature drift
    if feature_drift_summary.get('n_features_high_drift', 0) > 0:
        alerts.append({
            'severity': 'high',
            'category': 'feature_drift',
            'message': f"{feature_drift_summary['n_features_high_drift']} features showing significant drift",
            'recommendation': 'Investigate feature changes and data pipeline',
            'metrics': {
                'features': feature_drift_summary['features_high_drift']
            }
        })

    # Check data quality
    if data_quality_summary.get('max_missing_rate_pct', 0) > missing_rate_threshold:
        alerts.append({
            'severity': 'warning',
            'category': 'data_quality',
            'message': f"Missing value rate exceeds {missing_rate_threshold}% (max: {data_quality_summary['max_missing_rate_pct']:.2f}%)",
            'recommendation': 'Review data pipeline and investigate missing data sources'
        })

    return alerts


def generate_recommendations(alerts: List[Dict]) -> List[str]:
    """
    Generate actionable recommendations based on alerts

    Args:
        alerts: List of alerts from generate_alerts

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Count alerts by severity
    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
    high_alerts = [a for a in alerts if a['severity'] == 'high']
    warning_alerts = [a for a in alerts if a['severity'] == 'warning']

    # Add specific recommendations based on alerts
    for alert in alerts:
        if alert['recommendation'] not in recommendations:
            recommendations.append(alert['recommendation'])

    # Add general recommendations
    recommendations.append("Continue monitoring model performance monthly")
    recommendations.append("Set up automated alerts for PSI > 0.2 or feature drift > 0.2")
    recommendations.append("Schedule quarterly model performance reviews")

    # Prioritize critical actions
    if critical_alerts:
        recommendations.insert(0, f"URGENT: {len(critical_alerts)} critical issue(s) detected - immediate action required")

    return recommendations


# ============================================================================
# 6. VISUALIZATION AND REPORT GENERATION
# ============================================================================

def generate_monitoring_visualizations(
    psi_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    feature_drift_df: pd.DataFrame,
    data_quality_df: pd.DataFrame,
    all_predictions: pd.DataFrame,
    output_dir: str,
    metadata: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Generate comprehensive monitoring visualizations

    Args:
        psi_df: PSI results over time
        performance_df: Performance metrics over time
        feature_drift_df: Feature drift results
        data_quality_df: Data quality metrics
        all_predictions: All predictions data
        output_dir: Directory to save visualizations
        metadata: Optional model metadata

    Returns:
        Dictionary with paths to generated visualization files
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100

    visualization_paths = {}

    # 1. PSI Over Time
    if len(psi_df) > 0:
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

    # 2. Model Performance Dashboard (if available)
    if len(performance_df) > 0:
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

    # 3. Feature Drift Heatmap
    if len(feature_drift_df) > 0:
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

    # 4. Data Quality Trends
    if len(data_quality_df) > 0:
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

    # 5. Prediction Distribution
    if len(all_predictions) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Monthly statistics
        monthly_stats = all_predictions.groupby('inference_date').agg({
            'Customer_ID': 'count',
            'default_probability': 'mean',
            'predicted_default': 'mean'
        }).reset_index()
        monthly_stats.columns = ['date', 'n_customers', 'avg_prob', 'default_rate']

        # Customers scored
        axes[0].plot(monthly_stats['date'], monthly_stats['n_customers'],
                    marker='o', linewidth=2.5, markersize=7, color='#2E86AB')
        axes[0].set_title('Customers Scored per Month', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Number of Customers')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # Default rate
        axes[1].plot(monthly_stats['date'], monthly_stats['default_rate']*100,
                    marker='o', linewidth=2.5, markersize=7, color='#E63946')
        axes[1].set_title('Predicted Default Rate Over Time', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Default Rate (%)')
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        pred_path = f"{output_dir}/prediction_statistics.png"
        plt.savefig(pred_path, bbox_inches='tight')
        plt.close()
        visualization_paths['predictions'] = pred_path

    return visualization_paths


def generate_html_report(
    summary_report: Dict,
    visualization_paths: Dict[str, str],
    output_dir: str
) -> str:
    """
    Generate comprehensive HTML report with visualizations

    Args:
        summary_report: Monitoring summary from run_model_monitoring
        visualization_paths: Paths to visualization images
        output_dir: Directory to save HTML report

    Returns:
        Path to generated HTML report
    """
    import base64
    from datetime import datetime

    # Helper to encode images as base64
    def img_to_base64(img_path):
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        return ""

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
        .metric-card {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .alert {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 5px solid;
        }}
        .alert-critical {{
            background-color: #fee;
            border-color: #e74c3c;
            color: #c0392b;
        }}
        .alert-high {{
            background-color: #fef5e7;
            border-color: #f39c12;
            color: #d68910;
        }}
        .alert-warning {{
            background-color: #fff9e6;
            border-color: #f1c40f;
            color: #9a7d0a;
        }}
        .recommendation {{
            background-color: #e8f8f5;
            border-left: 4px solid #16a085;
            padding: 10px;
            margin: 5px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
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
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .viz-section {{
            margin: 30px 0;
            text-align: center;
        }}
        .viz-section img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
        }}
        .status-stable {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-drift {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Model Monitoring Report</h1>
        <p class="timestamp">Generated: {summary_report.get('timestamp', datetime.now().isoformat())}</p>

        <h2>1. Executive Summary</h2>
        <div style="text-align: center;">
"""

    # Add metric cards
    psi_summary = summary_report.get('psi_summary', {})
    perf_summary = summary_report.get('performance_summary', {})
    alerts = summary_report.get('alerts', [])

    critical_count = len([a for a in alerts if a['severity'] == 'critical'])

    html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{psi_summary.get('avg_psi', 0):.3f}</div>
                <div class="metric-label">Average PSI</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{psi_summary.get('n_significant', 0)}</div>
                <div class="metric-label">Periods with Drift</div>
            </div>
"""

    if perf_summary.get('evaluation_available'):
        avg_auc = perf_summary['avg_metrics']['auc']
        html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{avg_auc:.3f}</div>
                <div class="metric-label">Average AUC</div>
            </div>
"""

    html_content += f"""
            <div class="metric-card">
                <div class="metric-value">{len(alerts)}</div>
                <div class="metric-label">Total Alerts</div>
            </div>
        </div>
"""

    # Alerts section
    if alerts:
        html_content += """
        <h2>2. Alerts</h2>
"""
        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            html_content += f"""
        <div class="alert {severity_class}">
            <strong>[{alert['severity'].upper()}]</strong> {alert['message']}<br>
            <em>Recommendation: {alert['recommendation']}</em>
        </div>
"""

    # PSI Summary
    html_content += f"""
        <h2>3. Population Stability</h2>
        <p>
            <strong>Status:</strong>
            <span class="{'status-drift' if psi_summary.get('n_significant', 0) > 0 else 'status-stable'}">
                {'⚠️ DRIFT DETECTED' if psi_summary.get('n_significant', 0) > 0 else '✅ STABLE'}
            </span>
        </p>
        <ul>
            <li>Average PSI: <strong>{psi_summary.get('avg_psi', 0):.4f}</strong></li>
            <li>Max PSI: <strong>{psi_summary.get('max_psi', 0):.4f}</strong></li>
            <li>Stable periods (PSI < 0.1): <strong>{psi_summary.get('n_stable', 0)}</strong></li>
            <li>Moderate drift (0.1 ≤ PSI < 0.2): <strong>{psi_summary.get('n_moderate', 0)}</strong></li>
            <li>Significant drift (PSI ≥ 0.2): <strong>{psi_summary.get('n_significant', 0)}</strong></li>
        </ul>
"""

    if 'psi' in visualization_paths:
        psi_img = img_to_base64(visualization_paths['psi'])
        html_content += f"""
        <div class="viz-section">
            <img src="data:image/png;base64,{psi_img}" alt="PSI Over Time">
        </div>
"""

    # Performance Summary
    if perf_summary.get('evaluation_available'):
        html_content += """
        <h2>4. Model Performance</h2>
"""
        avg_metrics = perf_summary['avg_metrics']
        html_content += f"""
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr><td>AUC-ROC</td><td>{avg_metrics['auc']:.4f}</td></tr>
            <tr><td>Accuracy</td><td>{avg_metrics['accuracy']:.4f}</td></tr>
            <tr><td>Precision</td><td>{avg_metrics['precision']:.4f}</td></tr>
            <tr><td>Recall</td><td>{avg_metrics['recall']:.4f}</td></tr>
            <tr><td>F1-Score</td><td>{avg_metrics['f1_score']:.4f}</td></tr>
        </table>
"""

        if 'training_comparison' in perf_summary:
            comp = perf_summary['training_comparison']
            status_color = 'green' if comp['performance_status'] == 'better' else 'orange' if comp['performance_status'] == 'similar' else 'red'
            html_content += f"""
        <p><strong>Comparison with Training:</strong></p>
        <ul>
            <li>Training Test AUC: <strong>{comp['training_test_auc']:.4f}</strong></li>
            <li>Production Avg AUC: <strong>{comp['production_avg_auc']:.4f}</strong></li>
            <li>Difference: <strong style="color:{status_color}">{comp['auc_difference']:+.4f}</strong> ({comp['performance_status']})</li>
        </ul>
"""

        if 'performance' in visualization_paths:
            perf_img = img_to_base64(visualization_paths['performance'])
            html_content += f"""
        <div class="viz-section">
            <img src="data:image/png;base64,{perf_img}" alt="Performance Dashboard">
        </div>
"""

    # Feature Drift
    feat_summary = summary_report.get('feature_drift_summary', {})
    if feat_summary.get('n_features_monitored', 0) > 0:
        html_content += f"""
        <h2>5. Feature Drift</h2>
        <ul>
            <li>Features monitored: <strong>{feat_summary['n_features_monitored']}</strong></li>
            <li>Features with significant drift: <strong>{feat_summary.get('n_features_high_drift', 0)}</strong></li>
            <li>Average feature PSI: <strong>{feat_summary.get('avg_feature_psi', 0):.4f}</strong></li>
        </ul>
"""

        if 'feature_drift' in visualization_paths:
            drift_img = img_to_base64(visualization_paths['feature_drift'])
            html_content += f"""
        <div class="viz-section">
            <img src="data:image/png;base64,{drift_img}" alt="Feature Drift Heatmap">
        </div>
"""

    # Data Quality
    qual_summary = summary_report.get('data_quality_summary', {})
    if qual_summary:
        html_content += f"""
        <h2>6. Data Quality</h2>
        <ul>
            <li>Average missing rate: <strong>{qual_summary.get('avg_missing_rate_pct', 0):.2f}%</strong></li>
            <li>Max missing rate: <strong>{qual_summary.get('max_missing_rate_pct', 0):.2f}%</strong></li>
            <li>Average records per month: <strong>{qual_summary.get('avg_rows_per_month', 0):.0f}</strong></li>
        </ul>
"""

        if 'data_quality' in visualization_paths:
            qual_img = img_to_base64(visualization_paths['data_quality'])
            html_content += f"""
        <div class="viz-section">
            <img src="data:image/png;base64,{qual_img}" alt="Data Quality Trends">
        </div>
"""

    # Predictions
    if 'predictions' in visualization_paths:
        html_content += """
        <h2>7. Prediction Statistics</h2>
"""
        pred_img = img_to_base64(visualization_paths['predictions'])
        html_content += f"""
        <div class="viz-section">
            <img src="data:image/png;base64,{pred_img}" alt="Prediction Statistics">
        </div>
"""

    # Recommendations
    recommendations = summary_report.get('recommendations', [])
    if recommendations:
        html_content += """
        <h2>8. Recommendations</h2>
"""
        for i, rec in enumerate(recommendations, 1):
            html_content += f"""
        <div class="recommendation">
            <strong>{i}.</strong> {rec}
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    # Save HTML report
    report_path = f"{output_dir}/monitoring_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path


# ============================================================================
# 7. MAIN MONITORING WORKFLOW
# ============================================================================

def run_model_monitoring(config: Dict) -> Dict:
    """
    Main function to run comprehensive model monitoring

    Args:
        config: Configuration dictionary with keys:
            - predictions_dir: Path to predictions directory
            - model_dir: Path to model artifacts directory
            - feature_store_path: Path to feature store
            - label_store_path: Path to label store
            - output_dir: Path to save monitoring results
            - psi_threshold: PSI alert threshold (default: 0.2)
            - auc_drop_threshold: AUC drop threshold (default: 0.05)
            - missing_rate_threshold: Missing rate threshold (default: 5.0)

    Returns:
        Dictionary with monitoring results and alerts
    """
    print("="*80)
    print("STARTING MODEL MONITORING")
    print("="*80)

    # Initialize Spark
    spark = SparkSession.builder \
        .appName("model_monitoring") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # 1. Load model artifacts
        print("\n1. Loading model artifacts...")
        model_dir = config['model_dir']
        metadata = joblib.load(f"{model_dir}/model_metadata.pkl")
        feature_importance = pd.read_csv(f"{model_dir}/feature_importance.csv")
        print(f"   Model trained on: {metadata['training_date']}")

        # 2. Load predictions
        print("\n2. Loading predictions...")
        predictions_files = sorted(glob.glob(f"{config['predictions_dir']}/predictions_*.csv"))
        predictions_list = [pd.read_csv(f) for f in predictions_files]
        all_predictions = pd.concat(predictions_list, ignore_index=True)
        all_predictions['inference_date'] = pd.to_datetime(all_predictions['inference_date'])
        print(f"   Total predictions: {len(all_predictions):,}")
        print(f"   Date range: {all_predictions['inference_date'].min().date()} to {all_predictions['inference_date'].max().date()}")

        # 3. Calculate PSI
        print("\n3. Calculating Population Stability Index (PSI)...")
        psi_df = calculate_psi_over_time(all_predictions)
        psi_summary = get_psi_summary(psi_df)
        print(f"   Average PSI: {psi_summary['avg_psi']:.4f}")
        print(f"   Periods with significant drift: {psi_summary['n_significant']}")

        # 4. Calculate feature drift
        print("\n4. Calculating feature drift...")
        feature_drift_df = calculate_feature_drift(
            spark,
            config['feature_store_path'],
            feature_importance,
            top_n_features=10
        )
        feature_drift_summary = get_feature_drift_summary(feature_drift_df)
        print(f"   Features monitored: {feature_drift_summary['n_features_monitored']}")
        print(f"   Features with high drift: {feature_drift_summary['n_features_high_drift']}")

        # 5. Evaluate model performance
        print("\n5. Evaluating model performance...")
        performance_df = evaluate_model_performance(
            spark,
            all_predictions,
            config['label_store_path']
        )
        performance_summary = get_performance_summary(performance_df, metadata)
        if performance_summary.get('evaluation_available'):
            print(f"   Samples evaluated: {performance_summary['total_samples_evaluated']}")
            print(f"   Average AUC: {performance_summary['avg_metrics']['auc']:.4f}")
        else:
            print("   No labeled data available for evaluation")

        # 6. Monitor data quality
        print("\n6. Monitoring data quality...")
        data_quality_df = monitor_data_quality(spark, config['feature_store_path'])
        data_quality_summary = get_data_quality_summary(data_quality_df)
        print(f"   Average missing rate: {data_quality_summary['avg_missing_rate_pct']:.2f}%")
        print(f"   Max missing rate: {data_quality_summary['max_missing_rate_pct']:.2f}%")

        # 7. Generate alerts and recommendations
        print("\n7. Generating alerts and recommendations...")
        alerts = generate_alerts(
            psi_summary,
            performance_summary,
            feature_drift_summary,
            data_quality_summary,
            psi_threshold=config.get('psi_threshold', 0.2),
            auc_drop_threshold=config.get('auc_drop_threshold', 0.05),
            missing_rate_threshold=config.get('missing_rate_threshold', 5.0)
        )
        recommendations = generate_recommendations(alerts)
        print(f"   Alerts generated: {len(alerts)}")
        print(f"   Critical alerts: {len([a for a in alerts if a['severity'] == 'critical'])}")

        # 8. Generate visualizations
        print("\n8. Generating visualizations...")
        output_dir = config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        visualization_paths = generate_monitoring_visualizations(
            psi_df=psi_df,
            performance_df=performance_df,
            feature_drift_df=feature_drift_df,
            data_quality_df=data_quality_df,
            all_predictions=all_predictions,
            output_dir=output_dir,
            metadata=metadata
        )
        print(f"   Generated {len(visualization_paths)} visualizations")
        for viz_type, path in visualization_paths.items():
            print(f"     - {viz_type}: {os.path.basename(path)}")

        # 9. Save results
        print("\n9. Saving monitoring results...")

        psi_df.to_csv(f"{output_dir}/psi_results.csv", index=False)
        if len(feature_drift_df) > 0:
            feature_drift_df.to_csv(f"{output_dir}/feature_drift_results.csv", index=False)
        if len(performance_df) > 0:
            performance_df.to_csv(f"{output_dir}/performance_metrics.csv", index=False)
        data_quality_df.to_csv(f"{output_dir}/data_quality_metrics.csv", index=False)

        # Save summary report
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'psi_summary': psi_summary,
            'performance_summary': performance_summary,
            'feature_drift_summary': feature_drift_summary,
            'data_quality_summary': data_quality_summary,
            'alerts': alerts,
            'recommendations': recommendations
        }

        import json
        with open(f"{output_dir}/monitoring_summary.json", 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)

        print(f"   Results saved to: {output_dir}")

        print("\n" + "="*80)
        print("MODEL MONITORING COMPLETE")
        print("="*80)

        return summary_report

    finally:
        spark.stop()
        print("\nSpark session stopped")


# ============================================================================
# 7. HELPER FUNCTION TO CHECK IF RETRAINING IS NEEDED
# ============================================================================

def should_retrain_model(summary_report: Dict) -> Dict:
    """
    Determine if model retraining is needed based on monitoring results

    Args:
        summary_report: Output from run_model_monitoring

    Returns:
        Dictionary with recommendation and reasons
    """
    retrain = False
    reasons = []

    # Check for critical alerts
    critical_alerts = [a for a in summary_report['alerts'] if a['severity'] == 'critical']
    if critical_alerts:
        retrain = True
        for alert in critical_alerts:
            reasons.append(alert['message'])

    # Check PSI
    if summary_report['psi_summary']['n_significant'] >= 3:
        retrain = True
        reasons.append(f"Significant population drift in {summary_report['psi_summary']['n_significant']} periods")

    # Check performance degradation
    if 'training_comparison' in summary_report['performance_summary']:
        comp = summary_report['performance_summary']['training_comparison']
        if comp['performance_status'] == 'degraded' and comp['auc_difference'] < -0.05:
            retrain = True
            reasons.append(f"AUC dropped by {abs(comp['auc_difference']):.4f} from training baseline")

    return {
        'should_retrain': retrain,
        'reasons': reasons,
        'recommendation': 'Model retraining recommended' if retrain else 'Model is stable, no retraining needed'
    }
