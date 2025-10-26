"""
Model Inference Utilities for Credit Default Prediction

This module provides utilities for running inference with trained XGBoost models.
Adapted from CreditKarma reference implementation for XGBoost with custom preprocessing.
"""

import os
import glob
import pandas as pd
import numpy as np
import pyspark
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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


def load_features_for_inference(inference_date, gold_db, spark):
    """
    Load feature data for inference date

    Args:
        inference_date: Date string (YYYY-MM-DD) for inference
        gold_db: Path to gold database directory
        spark: Spark session

    Returns:
        Pandas DataFrame with features for all customers
    """
    print(f"\nLoading features for inference date: {inference_date}")

    # Read feature store
    X_spark = read_gold_table('feature_store', gold_db, spark)

    # Filter to inference date
    X_spark_filtered = X_spark.filter(X_spark.snapshot_date == inference_date)

    # Convert to pandas
    X_df = X_spark_filtered.toPandas()

    print(f"Loaded {len(X_df):,} customers with features")

    return X_df


##########################
# MODEL LOADING
##########################

def load_model_artifacts(model_dir='models'):
    """
    Load trained model and all artifacts

    Args:
        model_dir: Directory containing saved artifacts

    Returns:
        model, preprocessors, metadata
    """
    print(f"\nLoading model artifacts from: {model_dir}")

    # Load model
    model_path = os.path.join(model_dir, 'xgboost_credit_default_model.pkl')
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # Load preprocessors
    preprocessor_path = os.path.join(model_dir, 'preprocessors.pkl')
    preprocessors = joblib.load(preprocessor_path)
    print(f"Preprocessors loaded from: {preprocessor_path}")

    # Load metadata
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    metadata = joblib.load(metadata_path)
    print(f"Metadata loaded from: {metadata_path}")

    print(f"\nModel trained on: {metadata['training_date']}")
    print(f"Test AUC: {metadata['metrics']['test']['auc']:.4f}")
    print(f"OOT AUC: {metadata['metrics']['oot']['auc']:.4f}")

    return model, preprocessors, metadata


##########################
# PREPROCESSING
##########################

def preprocess_inference_data(df, preprocessors, exclude_cols):
    """
    Preprocess inference data using saved preprocessors

    Args:
        df: Raw feature DataFrame
        preprocessors: Dictionary of preprocessing objects
        exclude_cols: Columns to exclude from features

    Returns:
        X_processed: Preprocessed features ready for inference
        customer_ids: List of Customer_IDs (for joining predictions back)
    """
    print(f"\nPreprocessing inference data...")

    # Store customer IDs for later
    customer_ids = df['Customer_ID'].copy()

    # Get feature names from preprocessors
    numeric_features = preprocessors['numeric_features']
    categorical_features = preprocessors['categorical_features']
    all_features = preprocessors['feature_names']

    # Make a copy for processing
    df_proc = df.copy()

    # 1. Handle missing values - numeric (use training medians)
    numeric_medians = preprocessors['numeric_medians']
    for col in numeric_features:
        if col in df_proc.columns:
            median_val = numeric_medians.get(col, df_proc[col].median())
            df_proc[col].fillna(median_val, inplace=True)

    # 2. Encode categorical features (use training encoders)
    label_encoders = preprocessors['label_encoders']
    for col in categorical_features:
        if col in df_proc.columns:
            # Fill missing with 'Unknown'
            df_proc[col].fillna('Unknown', inplace=True)

            # Get encoder
            le = label_encoders[col]

            # Handle unseen categories by mapping to 'Unknown'
            df_proc[col] = df_proc[col].apply(
                lambda x: x if x in le.classes_ else 'Unknown'
            )

            # Transform
            df_proc[col] = le.transform(df_proc[col])

    # 3. Scale numeric features (use training scaler)
    scaler = preprocessors['scaler']
    df_proc[numeric_features] = scaler.transform(df_proc[numeric_features])

    # Extract features in correct order
    X_processed = df_proc[all_features]

    print(f"Preprocessed {len(X_processed):,} rows with {len(all_features)} features")

    return X_processed, customer_ids


##########################
# INFERENCE
##########################

def run_inference(model, X):
    """
    Run inference and get predictions

    Args:
        model: Trained model
        X: Preprocessed features

    Returns:
        predictions (0/1), prediction probabilities
    """
    print(f"\nRunning inference on {len(X):,} customers...")

    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Summary stats
    n_default = (y_pred == 1).sum()
    default_rate = n_default / len(y_pred) * 100

    print(f"Predicted defaults: {n_default:,} / {len(y_pred):,} ({default_rate:.1f}%)")
    print(f"Average default probability: {y_pred_proba.mean():.4f}")
    print(f"Probability range: [{y_pred_proba.min():.4f}, {y_pred_proba.max():.4f}]")

    return y_pred, y_pred_proba


##########################
# OUTPUT
##########################

def create_prediction_dataframe(customer_ids, predictions, probabilities, inference_date):
    """
    Create DataFrame with predictions

    Args:
        customer_ids: List of Customer_IDs
        predictions: Binary predictions (0/1)
        probabilities: Prediction probabilities
        inference_date: Date of inference

    Returns:
        DataFrame with predictions
    """
    df_predictions = pd.DataFrame({
        'Customer_ID': customer_ids,
        'inference_date': inference_date,
        'predicted_default': predictions,
        'default_probability': probabilities,
        'risk_category': pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
    })

    return df_predictions


def save_predictions(df_predictions, output_dir='predictions'):
    """
    Save predictions to file

    Args:
        df_predictions: DataFrame with predictions
        output_dir: Directory to save predictions

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    inference_date = df_predictions['inference_date'].iloc[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_{inference_date}_{timestamp}.csv'
    filepath = os.path.join(output_dir, filename)

    # Save
    df_predictions.to_csv(filepath, index=False)
    print(f"\nPredictions saved to: {filepath}")

    # Print summary
    print(f"\nPrediction Summary:")
    print(df_predictions['risk_category'].value_counts())

    return filepath


##########################
# MAIN FUNCTION
##########################

def predict_credit_default(config):
    """
    Main function to run credit default prediction inference

    Args:
        config: Configuration dictionary with:
            - inference_date: Date string (YYYY-MM-DD) for inference
            - gold_db: Path to gold database
            - model_dir: Directory with saved model artifacts
            - exclude_cols: Columns to exclude (should match training)
            - output_dir: Directory to save predictions

    Returns:
        df_predictions: DataFrame with predictions
    """
    print("="*80)
    print("CREDIT DEFAULT PREDICTION - INFERENCE")
    print("="*80)

    # Initialize Spark
    spark = pyspark.sql.SparkSession.builder \
        .appName("model-inference") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Load model artifacts
    print("\n1. Loading model artifacts...")
    model, preprocessors, metadata = load_model_artifacts(config['model_dir'])

    # Load features for inference
    print("\n2. Loading features...")
    X_df = load_features_for_inference(
        config['inference_date'],
        config['gold_db'],
        spark
    )

    if len(X_df) == 0:
        print(f"\nWARNING: No features found for date {config['inference_date']}")
        print("Please check if the date exists in feature_store")
        spark.stop()
        return None

    # Preprocess
    print("\n3. Preprocessing...")
    X_processed, customer_ids = preprocess_inference_data(
        X_df,
        preprocessors,
        config['exclude_cols']
    )

    # Run inference
    print("\n4. Running inference...")
    predictions, probabilities = run_inference(model, X_processed)

    # Create predictions DataFrame
    print("\n5. Creating prediction output...")
    df_predictions = create_prediction_dataframe(
        customer_ids,
        predictions,
        probabilities,
        config['inference_date']
    )

    # Save predictions
    print("\n6. Saving predictions...")
    output_path = save_predictions(
        df_predictions,
        output_dir=config.get('output_dir', 'predictions')
    )

    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"Scored {len(df_predictions):,} customers")
    print(f"Output: {output_path}")

    spark.stop()

    return df_predictions


##########################
# BATCH INFERENCE
##########################

def predict_credit_default_batch(config):
    """
    Run inference for multiple dates in batch

    Args:
        config: Configuration dictionary with:
            - inference_dates: List of date strings (YYYY-MM-DD)
            - gold_db: Path to gold database
            - model_dir: Directory with saved model artifacts
            - exclude_cols: Columns to exclude
            - output_dir: Directory to save predictions

    Returns:
        Dictionary mapping dates to prediction DataFrames
    """
    print("="*80)
    print("CREDIT DEFAULT PREDICTION - BATCH INFERENCE")
    print("="*80)
    print(f"\nProcessing {len(config['inference_dates'])} dates...")

    results = {}

    for inference_date in config['inference_dates']:
        print(f"\n{'='*80}")
        print(f"Processing: {inference_date}")
        print(f"{'='*80}")

        # Create config for single date
        date_config = config.copy()
        date_config['inference_date'] = inference_date

        # Run inference
        try:
            df_pred = predict_credit_default(date_config)
            results[inference_date] = df_pred
        except Exception as e:
            print(f"\nL ERROR processing {inference_date}: {str(e)}")
            results[inference_date] = None

    # Summary
    print("\n" + "="*80)
    print("BATCH INFERENCE COMPLETE")
    print("="*80)
    successful = len([r for r in results.values() if r is not None])
    print(f"Successfully processed: {successful}/{len(config['inference_dates'])} dates")

    return results
