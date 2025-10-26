"""
Model Training Utilities for Credit Default Prediction

This module provides utilities for training XGBoost models for credit default prediction.
"""

import os
import glob
import pandas as pd
import numpy as np
import pyspark
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
from xgboost import XGBClassifier

import joblib
import pickle


##########################
# DATA LOADING
##########################

def read_gold_table(table, gold_db, spark):
    """
    Helper function to read all partitions of a gold table

    Args:
        table: Table name ('feature_store' or 'label_store')
        gold_db: Path to gold database directory
        spark: Spark session

    Returns:
        Spark DataFrame with all partitions
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f))
                  for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").option("mergeSchema", "true").parquet(*files_list)
    return df


##########################
# PREPROCESSING
##########################

def merge_features_labels(X_df, y_df):
    """
    Merge features and labels on Customer_ID only

    Args:
        X_df: Features DataFrame
        y_df: Labels DataFrame

    Returns:
        Merged DataFrame
    """
    print(f"\nMerging features and labels...")
    print(f"Features: {X_df['Customer_ID'].nunique():,} unique customers")
    print(f"Labels: {y_df['Customer_ID'].nunique():,} unique customers")

    # Merge on Customer_ID only (NOT snapshot_date)
    df = y_df.merge(
        X_df,
        on='Customer_ID',
        how='inner',
        suffixes=('_label', '_feature')
    )

    # Rename columns
    df = df.rename(columns={
        'snapshot_date_label': 'label_date',
        'snapshot_date_feature': 'feature_date'
    })

    # Use label_date for temporal splitting
    df['snapshot_date'] = pd.to_datetime(df['label_date'])

    print(f"Merged: {len(df):,} records from {df['Customer_ID'].nunique():,} unique customers")

    return df


def split_dataset_temporal(df, config):
    """
    Split dataset temporally: train/val/test + OOT

    Args:
        df: Merged DataFrame with features and labels
        config: Configuration dictionary with split parameters

    Returns:
        train_df, val_df, test_df, oot_df
    """
    # Sort by date
    df = df.sort_values('snapshot_date').reset_index(drop=True)

    # Get unique dates
    unique_dates = sorted(df['snapshot_date'].unique())

    # Reserve last month for OOT
    oot_date = unique_dates[-1]
    dates_for_split = unique_dates[:-1]

    # Calculate split indices
    n_dates = len(dates_for_split)
    train_end_idx = int(n_dates * config['train_ratio'])
    val_end_idx = int(n_dates * (config['train_ratio'] + config['val_ratio']))

    # Get split dates
    train_dates = dates_for_split[:train_end_idx]
    val_dates = dates_for_split[train_end_idx:val_end_idx]
    test_dates = dates_for_split[val_end_idx:]

    # Create splits
    train_df = df[df['snapshot_date'].isin(train_dates)].copy()
    val_df = df[df['snapshot_date'].isin(val_dates)].copy()
    test_df = df[df['snapshot_date'].isin(test_dates)].copy()
    oot_df = df[df['snapshot_date'] == oot_date].copy()

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df):,} rows ({len(train_dates)} months)")
    print(f"  Val: {len(val_df):,} rows ({len(val_dates)} months)")
    print(f"  Test: {len(test_df):,} rows ({len(test_dates)} months)")
    print(f"  OOT: {len(oot_df):,} rows (1 month)")

    return train_df, val_df, test_df, oot_df


def preprocess_features(train_df, val_df, test_df, oot_df, exclude_cols, categorical_features):
    """
    Preprocess features: handle missing values, encode categoricals, scale numerics

    Args:
        train_df, val_df, test_df, oot_df: Data splits
        exclude_cols: Columns to exclude from features
        categorical_features: List of categorical feature names

    Returns:
        X_train, X_val, X_test, X_oot, y_train, y_val, y_test, y_oot, preprocessors
    """
    # Identify feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    numeric_features = [col for col in feature_cols if col not in categorical_features]

    print(f"\nPreprocessing features...")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numeric: {len(numeric_features)}")
    print(f"  Categorical: {len(categorical_features)}")

    # Make copies
    train_proc = train_df.copy()
    val_proc = val_df.copy()
    test_proc = test_df.copy()
    oot_proc = oot_df.copy()

    # 1. Handle missing values - numeric (median)
    numeric_medians = {}
    for col in numeric_features:
        median_val = train_proc[col].median()
        numeric_medians[col] = median_val
        for df_proc in [train_proc, val_proc, test_proc, oot_proc]:
            df_proc[col].fillna(median_val, inplace=True)

    # 2. Encode categorical features
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()

        # Fill missing with 'Unknown'
        for df_proc in [train_proc, val_proc, test_proc, oot_proc]:
            df_proc[col].fillna('Unknown', inplace=True)

        # Fit on training
        le.fit(train_proc[col])
        label_encoders[col] = le

        # Transform all
        train_proc[col] = le.transform(train_proc[col])

        # Handle unseen categories
        for df_proc in [val_proc, test_proc, oot_proc]:
            df_proc[col] = df_proc[col].apply(
                lambda x: x if x in le.classes_ else 'Unknown'
            )
            df_proc[col] = le.transform(df_proc[col])

    # 3. Scale numeric features
    scaler = StandardScaler()
    scaler.fit(train_proc[numeric_features])

    for df_proc in [train_proc, val_proc, test_proc, oot_proc]:
        df_proc[numeric_features] = scaler.transform(df_proc[numeric_features])

    # Create X and y
    all_features = numeric_features + categorical_features

    X_train = train_proc[all_features]
    X_val = val_proc[all_features]
    X_test = test_proc[all_features]
    X_oot = oot_proc[all_features]

    y_train = train_proc['label']
    y_val = val_proc['label']
    y_test = test_proc['label']
    y_oot = oot_proc['label']

    # Store preprocessors
    preprocessors = {
        'numeric_medians': numeric_medians,
        'label_encoders': label_encoders,
        'scaler': scaler,
        'feature_names': all_features,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }

    return X_train, X_val, X_test, X_oot, y_train, y_val, y_test, y_oot, preprocessors


##########################
# TRAINING
##########################

def train_xgboost_with_cv(X_train, y_train, X_val, y_val, config):
    """
    Train XGBoost with RandomizedSearchCV hyperparameter tuning

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        config: Configuration dictionary

    Returns:
        best_model, best_params, random_search object, scale_pos_weight
    """
    # Calculate class weights
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"\nClass imbalance: {n_neg} negative, {n_pos} positive")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Define search space
    param_distributions = config.get('param_distributions', {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 1.5, 2, 3]
    })

    # Create model (without early_stopping for CV)
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    # RandomizedSearchCV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=config.get('n_iter', 50),
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    print(f"\nStarting hyperparameter search...")
    random_search.fit(X_train, y_train)

    print(f"\nBest CV AUC: {random_search.best_score_:.4f}")
    print(f"Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")

    # Retrain with early stopping
    best_model = XGBClassifier(
        **random_search.best_params_,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=10
    )

    best_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"\nFinal model retrained with early stopping")
    print(f"Best iteration: {best_model.best_iteration}")

    return best_model, random_search.best_params_, random_search, scale_pos_weight


def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate model and return metrics

    Args:
        model: Trained model
        X, y: Test data
        dataset_name: Name for logging

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'auc': roc_auc_score(y, y_pred_proba)
    }

    print(f"\n{dataset_name} Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics, y_pred, y_pred_proba


def save_model_artifacts(model, preprocessors, best_params, metrics_dict, config, output_dir='models'):
    """
    Save model and all artifacts

    Args:
        model: Trained model
        preprocessors: Preprocessing objects
        best_params: Best hyperparameters
        metrics_dict: Dictionary of metrics for all splits
        config: Training configuration
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, 'xgboost_credit_default_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save preprocessors
    preprocessor_path = os.path.join(output_dir, 'preprocessors.pkl')
    joblib.dump(preprocessors, preprocessor_path)
    print(f"Preprocessors saved to: {preprocessor_path}")

    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'training_date': str(datetime.now()),
        'best_params': best_params,
        'metrics': metrics_dict,
        'config': config
    }

    metadata_path = os.path.join(output_dir, 'model_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saved to: {metadata_path}")

    # Save feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': preprocessors['feature_names'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        feat_imp_path = os.path.join(output_dir, 'feature_importance.csv')
        feature_importance.to_csv(feat_imp_path, index=False)
        print(f"Feature importance saved to: {feat_imp_path}")


##########################
# MAIN FUNCTION
##########################

def train_credit_default_model(config):
    """
    Main function to train credit default prediction model

    Args:
        config: Configuration dictionary with:
            - gold_db: Path to gold database
            - train_ratio: Train split ratio (default 0.70)
            - val_ratio: Val split ratio (default 0.15)
            - exclude_cols: Columns to exclude
            - categorical_features: List of categorical features
            - param_distributions: XGBoost hyperparameter search space
            - n_iter: Number of random search iterations
            - output_dir: Directory to save artifacts

    Returns:
        model, preprocessors, metrics
    """
    print("="*80)
    print("CREDIT DEFAULT PREDICTION - MODEL TRAINING")
    print("="*80)

    # Initialize Spark
    spark = pyspark.sql.SparkSession.builder \
        .appName("model-training") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Load data
    print("\n1. Loading data...")
    X_spark = read_gold_table('feature_store', config['gold_db'], spark)
    y_spark = read_gold_table('label_store', config['gold_db'], spark)

    X_df = X_spark.toPandas()
    y_df = y_spark.toPandas()

    # Merge
    print("\n2. Merging features and labels...")
    df = merge_features_labels(X_df, y_df)

    # Split
    print("\n3. Creating temporal splits...")
    train_df, val_df, test_df, oot_df = split_dataset_temporal(df, config)

    # Preprocess
    print("\n4. Preprocessing...")
    X_train, X_val, X_test, X_oot, y_train, y_val, y_test, y_oot, preprocessors = preprocess_features(
        train_df, val_df, test_df, oot_df,
        config['exclude_cols'],
        config['categorical_features']
    )

    # Train
    print("\n5. Training XGBoost model...")
    model, best_params, random_search, scale_pos_weight = train_xgboost_with_cv(
        X_train, y_train, X_val, y_val, config
    )

    # Evaluate
    print("\n6. Evaluating model...")
    metrics_train, _, _ = evaluate_model(model, X_train, y_train, "Train")
    metrics_val, _, _ = evaluate_model(model, X_val, y_val, "Validation")
    metrics_test, _, _ = evaluate_model(model, X_test, y_test, "Test")
    metrics_oot, _, _ = evaluate_model(model, X_oot, y_oot, "OOT")

    metrics_dict = {
        'train': metrics_train,
        'val': metrics_val,
        'test': metrics_test,
        'oot': metrics_oot,
        'scale_pos_weight': scale_pos_weight
    }

    # Save
    print("\n7. Saving artifacts...")
    save_model_artifacts(
        model, preprocessors, best_params, metrics_dict, config,
        output_dir=config.get('output_dir', 'models')
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Test AUC: {metrics_test['auc']:.4f}")
    print(f"OOT AUC: {metrics_oot['auc']:.4f}")

    spark.stop()

    return model, preprocessors, metrics_dict
