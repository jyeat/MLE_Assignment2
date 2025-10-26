# Model Monitoring - Evaluation Metrics & Business Metrics Supplement

Add these cells to your [model_monitoring.ipynb](model_monitoring.ipynb) after **Section 3.3 (Score Distribution Analysis)** to track actual model performance and business impact.

---

## Section 3.4: Model Evaluation Metrics (with Actual Labels)

### Cell 1: Markdown Header

```markdown
### 3.4 Model Evaluation Metrics (with Actual Labels)

**Key Performance Metrics**:
- **AUC-ROC**: Ability to distinguish between default and non-default
- **Precision**: Of predicted defaults, how many actually defaulted
- **Recall**: Of actual defaults, how many did we catch
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness

**Why These Metrics Matter**:
- **AUC**: Measures model's ranking ability (higher = better discrimination)
- **Precision**: Business cost of false alarms (rejecting good customers)
- **Recall**: Business cost of missed defaults (lending to bad customers)
- **F1-Score**: Balance between precision and recall
```

### Cell 2: Load Labels

```python
# Load label store to get actual outcomes
print("Loading label store for performance evaluation...")

label_path = '/app/datamart/gold/label_store/'

try:
    # Load all label files
    df_labels_spark = spark.read.parquet(f'{label_path}*/*.parquet')
    df_labels = df_labels_spark.toPandas()
    df_labels['label_date'] = pd.to_datetime(df_labels['label_date'])

    print(f"✅ Label store loaded: {len(df_labels):,} rows")
    print(f"Date range: {df_labels['label_date'].min()} to {df_labels['label_date'].max()}")
    print(f"\nLabel distribution:")
    print(df_labels['label'].value_counts())

    # Flag to indicate labels are available
    labels_available = True

except Exception as e:
    print(f"⚠️  Could not load labels: {e}")
    print("Skipping evaluation metrics that require actual labels")
    labels_available = False
```

### Cell 3: Merge Predictions with Labels

```python
# Merge predictions with actual labels (if available)
if labels_available:
    print("\nMerging predictions with actual outcomes...")

    # For monitoring, we match predictions with labels by Customer_ID and approximate date
    # In production, you'd match on loan_id and exact observation windows

    # Create merged dataset
    df_eval = all_predictions.merge(
        df_labels[['Customer_ID', 'label', 'label_date']],
        on='Customer_ID',
        how='inner'
    )

    # Filter to cases where label date is within reasonable window of inference date
    # (e.g., label observed within 6 months after prediction)
    df_eval['date_diff_months'] = (
        (df_eval['label_date'].dt.year - df_eval['inference_date'].dt.year) * 12 +
        (df_eval['label_date'].dt.month - df_eval['inference_date'].dt.month)
    )

    # Keep predictions where label came 0-6 months after prediction
    df_eval = df_eval[(df_eval['date_diff_months'] >= 0) & (df_eval['date_diff_months'] <= 6)]

    print(f"✅ Merged dataset: {len(df_eval):,} predictions with labels")
    print(f"Unique customers: {df_eval['Customer_ID'].nunique():,}")

    if len(df_eval) > 0:
        # Calculate metrics by month
        monthly_metrics = []

        for month in sorted(df_eval['inference_date'].unique()):
            month_data = df_eval[df_eval['inference_date'] == month]

            if len(month_data) > 0 and month_data['label'].nunique() > 1:
                y_true = month_data['label']
                y_pred = month_data['predicted_default']
                y_prob = month_data['default_probability']

                # Calculate metrics
                auc = roc_auc_score(y_true, y_prob)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                # Confusion matrix components
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

                # Business metrics
                actual_default_rate = y_true.mean()
                predicted_default_rate = y_pred.mean()

                # False Negative Rate (missed defaults)
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                # False Positive Rate (wrongly flagged as default)
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

        metrics_df = pd.DataFrame(monthly_metrics)

        print("\n" + "="*100)
        print("MONTHLY MODEL EVALUATION METRICS")
        print("="*100)
        print(metrics_df[['month', 'n_samples', 'auc', 'accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False))
    else:
        print("⚠️  No predictions with matching labels found")
        metrics_df = pd.DataFrame()
else:
    metrics_df = pd.DataFrame()
```

### Cell 4: Visualize Performance Metrics

```python
# Plot model evaluation metrics over time
if len(metrics_df) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. AUC over time
    axes[0, 0].plot(metrics_df['month'], metrics_df['auc'], marker='o', linewidth=2, markersize=6, color='blue')
    axes[0, 0].axhline(y=metadata['test_auc'], color='red', linestyle='--', linewidth=2, label=f'Test AUC: {metadata["test_auc"]:.3f}')
    axes[0, 0].axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (0.7)')
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.8)')
    axes[0, 0].set_title('AUC-ROC Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0.5, 1.0)

    # 2. Precision and Recall
    axes[0, 1].plot(metrics_df['month'], metrics_df['precision'], marker='o', linewidth=2, markersize=6, label='Precision')
    axes[0, 1].plot(metrics_df['month'], metrics_df['recall'], marker='s', linewidth=2, markersize=6, label='Recall')
    axes[0, 1].set_title('Precision vs Recall Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1.0)

    # 3. F1-Score
    axes[0, 2].plot(metrics_df['month'], metrics_df['f1_score'], marker='o', linewidth=2, markersize=6, color='purple')
    axes[0, 2].set_title('F1-Score Over Time', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Month')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].set_ylim(0, 1.0)

    # 4. Accuracy
    axes[1, 0].plot(metrics_df['month'], metrics_df['accuracy'], marker='o', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_title('Accuracy Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0.5, 1.0)

    # 5. Actual vs Predicted Default Rate
    axes[1, 1].plot(metrics_df['month'], metrics_df['actual_default_rate'] * 100,
                    marker='o', linewidth=2, markersize=6, label='Actual', color='red')
    axes[1, 1].plot(metrics_df['month'], metrics_df['predicted_default_rate'] * 100,
                    marker='s', linewidth=2, markersize=6, label='Predicted', color='blue')
    axes[1, 1].set_title('Actual vs Predicted Default Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Default Rate (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. False Positive vs False Negative Rate
    axes[1, 2].plot(metrics_df['month'], metrics_df['false_positive_rate'] * 100,
                    marker='o', linewidth=2, markersize=6, label='False Positive Rate', color='orange')
    axes[1, 2].plot(metrics_df['month'], metrics_df['false_negative_rate'] * 100,
                    marker='s', linewidth=2, markersize=6, label='False Negative Rate', color='red')
    axes[1, 2].set_title('Error Rates Over Time', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Month')
    axes[1, 2].set_ylabel('Error Rate (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    print("\n✅ Model evaluation metrics visualized")

    # Performance degradation check
    print("\n" + "="*100)
    print("PERFORMANCE DEGRADATION ANALYSIS")
    print("="*100)

    # Compare to training metrics
    current_avg_auc = metrics_df['auc'].mean()
    auc_degradation = metadata['test_auc'] - current_avg_auc

    print(f"Training Test AUC: {metadata['test_auc']:.4f}")
    print(f"Current Average AUC: {current_avg_auc:.4f}")
    print(f"AUC Degradation: {auc_degradation:.4f} ({auc_degradation/metadata['test_auc']*100:.1f}%)")

    if auc_degradation > 0.05:
        print("\n⚠️  WARNING: Significant AUC degradation detected (>5%)")
        print("Recommendation: Investigate root causes and consider model retraining")
    elif auc_degradation > 0.02:
        print("\n⚠️  CAUTION: Moderate AUC degradation detected (2-5%)")
        print("Recommendation: Monitor closely")
    else:
        print("\n✅ AUC performance is stable")

else:
    print("\n⚠️  Skipping evaluation metrics visualization (no labels available)")
```

---

## Section 3.5: Business Metrics Tracking

### Cell 5: Markdown Header

```markdown
### 3.5 Business Metrics Tracking

**Key Business Metrics**:
1. **Expected Loss**: Average loan amount × Default probability × (1 - Recovery rate)
2. **Approval Rate**: % of applications approved (predicted_default = 0)
3. **Default Capture Rate**: % of actual defaults correctly identified
4. **Precision at Top K**: Precision among highest-risk predictions
5. **Cost-Benefit Analysis**: Financial impact of model decisions

**Model Metrics → Business Impact Mapping**:
- **High Recall** → Catch more defaults → Reduce losses
- **High Precision** → Fewer false alarms → Approve more good customers → More revenue
- **High AUC** → Better rank ordering → Optimize approval thresholds
- **Low FNR** → Fewer missed defaults → Lower credit losses
- **Low FPR** → Fewer rejected good customers → Higher revenue
```

### Cell 6: Calculate Business Metrics

```python
# Calculate business metrics
print("="*100)
print("BUSINESS METRICS TRACKING")
print("="*100)

# Business assumptions (adjust based on actual business context)
AVERAGE_LOAN_AMOUNT = 10000  # Average loan amount in dollars
RECOVERY_RATE = 0.30  # Average recovery rate on defaulted loans (30%)
LOSS_GIVEN_DEFAULT = 1 - RECOVERY_RATE  # 70% loss on defaults
REVENUE_PER_LOAN = 500  # Average revenue (interest) per non-default loan

print("\nBusiness Assumptions:")
print(f"  Average Loan Amount: ${AVERAGE_LOAN_AMOUNT:,}")
print(f"  Recovery Rate on Defaults: {RECOVERY_RATE*100:.0f}%")
print(f"  Loss Given Default: {LOSS_GIVEN_DEFAULT*100:.0f}%")
print(f"  Revenue per Good Loan: ${REVENUE_PER_LOAN:,}")

if len(metrics_df) > 0:
    business_metrics = []

    for _, row in metrics_df.iterrows():
        month = row['month']
        month_data = df_eval[df_eval['inference_date'] == month]

        if len(month_data) > 0:
            # 1. Expected Loss
            # Expected Loss = Sum(Loan Amount × Default Probability × Loss Given Default)
            expected_loss = (month_data['default_probability'] * AVERAGE_LOAN_AMOUNT * LOSS_GIVEN_DEFAULT).sum()
            expected_loss_per_customer = expected_loss / len(month_data)

            # 2. Approval Rate (% of loans approved, i.e., predicted as non-default)
            approval_rate = 1 - row['predicted_default_rate']

            # 3. Default Capture Rate (Recall)
            default_capture_rate = row['recall']

            # 4. Precision at Top 10% (highest risk)
            top_10_pct = int(len(month_data) * 0.10)
            if top_10_pct > 0:
                top_risk = month_data.nlargest(top_10_pct, 'default_probability')
                precision_at_10 = top_risk['label'].mean() if len(top_risk) > 0 else 0
            else:
                precision_at_10 = 0

            # 5. Cost-Benefit Analysis
            # Cost = False Negatives (missed defaults) × Loan Amount × Loss Given Default
            cost_fn = row['false_negatives'] * AVERAGE_LOAN_AMOUNT * LOSS_GIVEN_DEFAULT

            # Cost = False Positives (rejected good customers) × Revenue Lost
            cost_fp = row['false_positives'] * REVENUE_PER_LOAN

            # Benefit = True Negatives (approved good customers) × Revenue
            benefit_tn = row['true_negatives'] * REVENUE_PER_LOAN

            # Benefit = True Positives (rejected bad customers) × Loss Avoided
            benefit_tp = row['true_positives'] * AVERAGE_LOAN_AMOUNT * LOSS_GIVEN_DEFAULT

            total_cost = cost_fn + cost_fp
            total_benefit = benefit_tn + benefit_tp
            net_benefit = total_benefit - total_cost

            # ROI
            roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0

            business_metrics.append({
                'month': month,
                'expected_loss': expected_loss,
                'expected_loss_per_customer': expected_loss_per_customer,
                'approval_rate': approval_rate,
                'default_capture_rate': default_capture_rate,
                'precision_at_top_10pct': precision_at_10,
                'cost_false_negatives': cost_fn,
                'cost_false_positives': cost_fp,
                'total_cost': total_cost,
                'total_benefit': total_benefit,
                'net_benefit': net_benefit,
                'roi_percent': roi
            })

    business_metrics_df = pd.DataFrame(business_metrics)

    print("\n" + "="*100)
    print("MONTHLY BUSINESS METRICS")
    print("="*100)
    print(business_metrics_df[[
        'month', 'expected_loss', 'approval_rate',
        'default_capture_rate', 'precision_at_top_10pct'
    ]].to_string(index=False))

    print("\n" + "="*100)
    print("FINANCIAL IMPACT")
    print("="*100)
    print(business_metrics_df[[
        'month', 'total_cost', 'total_benefit', 'net_benefit', 'roi_percent'
    ]].to_string(index=False))
else:
    business_metrics_df = pd.DataFrame()
```

### Cell 7: Visualize Business Metrics

```python
# Plot business metrics
if len(business_metrics_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Expected Loss per Customer
    axes[0, 0].plot(business_metrics_df['month'], business_metrics_df['expected_loss_per_customer'],
                    marker='o', linewidth=2, markersize=6, color='red')
    axes[0, 0].set_title('Expected Loss per Customer Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Expected Loss ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Approval Rate vs Default Capture Rate (Trade-off)
    ax2 = axes[0, 1]
    ax2.plot(business_metrics_df['month'], business_metrics_df['approval_rate'] * 100,
             marker='o', linewidth=2, markersize=6, color='green', label='Approval Rate')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Approval Rate (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(business_metrics_df['month'], business_metrics_df['default_capture_rate'] * 100,
                  marker='s', linewidth=2, markersize=6, color='red', label='Default Capture Rate')
    ax2_twin.set_ylabel('Default Capture Rate (%)', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')

    axes[0, 1].set_title('Approval Rate vs Default Capture Rate', fontsize=12, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # 3. Precision at Top 10%
    axes[1, 0].plot(business_metrics_df['month'], business_metrics_df['precision_at_top_10pct'] * 100,
                    marker='o', linewidth=2, markersize=6, color='purple')
    axes[1, 0].set_title('Precision at Top 10% Highest Risk', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Precision (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Net Benefit (Financial Impact)
    colors = ['green' if x > 0 else 'red' for x in business_metrics_df['net_benefit']]
    axes[1, 1].bar(range(len(business_metrics_df)), business_metrics_df['net_benefit'], color=colors, alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_title('Net Financial Benefit Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Net Benefit ($)')
    axes[1, 1].set_xticks(range(len(business_metrics_df)))
    axes[1, 1].set_xticklabels([m.strftime('%Y-%m') for m in business_metrics_df['month']], rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\n" + "="*100)
    print("BUSINESS METRICS SUMMARY")
    print("="*100)
    print(f"Total Expected Loss: ${business_metrics_df['expected_loss'].sum():,.2f}")
    print(f"Average Approval Rate: {business_metrics_df['approval_rate'].mean()*100:.2f}%")
    print(f"Average Default Capture Rate: {business_metrics_df['default_capture_rate'].mean()*100:.2f}%")
    print(f"Average Precision at Top 10%: {business_metrics_df['precision_at_top_10pct'].mean()*100:.2f}%")
    print(f"\nTotal Cost (FN + FP): ${business_metrics_df['total_cost'].sum():,.2f}")
    print(f"Total Benefit (TP + TN): ${business_metrics_df['total_benefit'].sum():,.2f}")
    print(f"Net Benefit: ${business_metrics_df['net_benefit'].sum():,.2f}")
    print(f"Average ROI: {business_metrics_df['roi_percent'].mean():.2f}%")

    print("\n✅ Business metrics visualized")

    # Save business metrics
    business_metrics_df.to_csv('/app/monitoring_reports/business_metrics.csv', index=False)
    print("\n✅ Business metrics saved to: /app/monitoring_reports/business_metrics.csv")

else:
    print("\n⚠️  Skipping business metrics (no evaluation data available)")
```

### Cell 8: Model-to-Business Metrics Mapping

```python
# Create a comprehensive model-to-business metrics mapping table
if len(metrics_df) > 0 and len(business_metrics_df) > 0:
    print("\n" + "="*100)
    print("MODEL METRICS → BUSINESS IMPACT MAPPING")
    print("="*100)

    # Combine metrics for correlation analysis
    combined_df = metrics_df.merge(business_metrics_df, on='month')

    # Key relationships
    print("\n1. RECALL (Default Capture Rate) → Expected Loss")
    print("   Higher Recall = More defaults caught = Lower expected loss")
    correlation_recall_loss = combined_df[['recall', 'expected_loss_per_customer']].corr().iloc[0, 1]
    print(f"   Correlation: {correlation_recall_loss:.3f}")

    print("\n2. PRECISION → Approval Rate")
    print("   Higher Precision = Fewer false alarms = More approvals")
    correlation_precision_approval = combined_df[['precision', 'approval_rate']].corr().iloc[0, 1]
    print(f"   Correlation: {correlation_precision_approval:.3f}")

    print("\n3. AUC → Net Benefit")
    print("   Higher AUC = Better risk ranking = Higher profitability")
    correlation_auc_benefit = combined_df[['auc', 'net_benefit']].corr().iloc[0, 1]
    print(f"   Correlation: {correlation_auc_benefit:.3f}")

    print("\n4. FALSE NEGATIVE RATE → Cost of Missed Defaults")
    print("   Higher FNR = More missed defaults = Higher losses")
    correlation_fnr_cost = combined_df[['false_negative_rate', 'cost_false_negatives']].corr().iloc[0, 1]
    print(f"   Correlation: {correlation_fnr_cost:.3f}")

    # Create summary table
    summary_table = pd.DataFrame({
        'Model Metric': ['AUC', 'Precision', 'Recall', 'F1-Score'],
        'Average Value': [
            f"{metrics_df['auc'].mean():.3f}",
            f"{metrics_df['precision'].mean():.3f}",
            f"{metrics_df['recall'].mean():.3f}",
            f"{metrics_df['f1_score'].mean():.3f}"
        ],
        'Business Impact': [
            f"ROI: {business_metrics_df['roi_percent'].mean():.1f}%",
            f"Approval Rate: {business_metrics_df['approval_rate'].mean()*100:.1f}%",
            f"Default Capture: {business_metrics_df['default_capture_rate'].mean()*100:.1f}%",
            f"Net Benefit: ${business_metrics_df['net_benefit'].mean():,.0f}/month"
        ],
        'Status': [
            '✅ Stable' if metrics_df['auc'].std() < 0.05 else '⚠️ Volatile',
            '✅ Stable' if metrics_df['precision'].std() < 0.05 else '⚠️ Volatile',
            '✅ Stable' if metrics_df['recall'].std() < 0.05 else '⚠️ Volatile',
            '✅ Stable' if metrics_df['f1_score'].std() < 0.05 else '⚠️ Volatile'
        ]
    })

    print("\n" + "="*100)
    print("MODEL-BUSINESS PERFORMANCE SUMMARY")
    print("="*100)
    print(summary_table.to_string(index=False))

    print("\n✅ Model-to-business mapping complete")
```

---

## Instructions

1. **Open** [model_monitoring.ipynb](model_monitoring.ipynb) in Jupyter Lab
2. **Insert** these cells after Section 3.3 (Score Distribution Analysis)
3. **Run** the cells in order
4. **Adjust** business assumptions (loan amount, recovery rate, revenue) to match your actual business

## Key Insights You'll Get

**Model Performance**:
- Is my model's AUC degrading over time?
- Are precision/recall maintaining acceptable levels?
- How well does predicted default rate match actual default rate?

**Business Impact**:
- What's my expected financial loss per month?
- What's the trade-off between approval rate and default capture?
- What's my ROI on the credit model?
- How much am I losing from false negatives vs false positives?

**Action Triggers**:
- AUC drops > 5% → Retrain model
- Expected loss increases > 20% → Review approval thresholds
- Precision at top 10% < 50% → Investigate feature drift
- Net benefit turns negative → Urgent model review
