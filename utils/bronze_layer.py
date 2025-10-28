import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_clickstream(snapshot_date_str,bronze_lms_directory,spark):

    snapshot_date=datetime.strptime(snapshot_date_str,"%Y-%m-%d")

    csv_file_path="data/feature_clickstream.csv"

    df=spark.read.csv(csv_file_path,header=True,inferSchema=True).filter(col('snapshot_date')==snapshot_date)
    print(snapshot_date_str + 'row count:',df.count())

    partition_name="bronze_clickstream_daily_"+ snapshot_date_str.replace('-','_')+'.csv'
    filepath=bronze_lms_directory+partition_name
    df.toPandas().to_csv(filepath,index=False)
    print('saved to:',filepath) 

    return df       

def process_bronze_attributes(snapshot_date_str,bronze_lms_directory,spark):

    snapshot_date=datetime.strptime(snapshot_date_str,"%Y-%m-%d")

    csv_file_path="data/features_attributes.csv"

    df=spark.read.csv(csv_file_path,header=True,inferSchema=True).filter(col('snapshot_date')==snapshot_date)
    print(snapshot_date_str + 'row count:',df.count())

    partition_name="bronze_attributes_daily_"+ snapshot_date_str.replace('-','_')+'.csv'
    filepath=bronze_lms_directory+partition_name
    df.toPandas().to_csv(filepath,index=False)
    print('saved to:',filepath) 

    return df       

def process_bronze_financials(snapshot_date_str,bronze_lms_directory,spark):
    
    snapshot_date=datetime.strptime(snapshot_date_str,'%Y-%m-%d') 
    csv_file_path="data/features_financials.csv"

    df=spark.read.csv(csv_file_path,header=True,inferSchema=True).filter(col('snapshot_date')==snapshot_date)
    print(snapshot_date_str+'row count:',df.count())

    partition_name="bronze_financials_daily_"+snapshot_date_str.replace('-','_')+'.csv'
    filepath=bronze_lms_directory+partition_name
    df.toPandas().to_csv(filepath,index=False)
    print('saved to:',filepath)

    return df

def process_bronze_loan(snapshot_date_str,bronze_lms_directory,spark):
    snapshot_date=datetime.strptime(snapshot_date_str,'%Y-%m-%d')
    csv_file_path='data/lms_loan_daily.csv'

    df=spark.read.csv(csv_file_path,header=True,inferSchema=True).filter(col('snapshot_date')==snapshot_date)
    print(snapshot_date_str +'row count:',df.count())

    partition_name="bronze_loan_daily_"+snapshot_date_str.replace('-','_')+'.csv'
    filepath=bronze_lms_directory+partition_name
    df.toPandas().to_csv(filepath,index=False)
    print('saved to:',filepath)

    return df

def process_bronze_table_main(table_name, source, bronze_db, snapshot_date_str):
    """
    Main function to process any bronze table
    Called by Airflow DAG with op_kwargs

    Args:
        table_name: 'clickstream', 'attributes', 'financials', or 'lms'
        source: Path to source CSV file (not used, but kept for compatibility)
        bronze_db: Path to bronze database directory
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
    """
    import pyspark

    # Initialize spark to None for proper cleanup
    spark = None

    try:
        # Create Spark session
        spark = pyspark.sql.SparkSession.builder \
            .appName("bronze-pipeline") \
            .master("local[*]") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")

        # Make sure bronze_db ends with '/'
        if not bronze_db.endswith('/'):
            bronze_db = bronze_db + '/'

        # Route to correct function based on table_name
        if table_name == 'clickstream':
            process_bronze_clickstream(snapshot_date_str, bronze_db, spark)
        elif table_name == 'attributes':
            process_bronze_attributes(snapshot_date_str, bronze_db, spark)
        elif table_name == 'financials':
            process_bronze_financials(snapshot_date_str, bronze_db, spark)
        elif table_name == 'lms':
            process_bronze_loan(snapshot_date_str, bronze_db, spark)
        else:
            raise ValueError(f"Unknown table_name: {table_name}")

        print(f"✅ Bronze {table_name} completed for {snapshot_date_str}")

    finally:
        # Always clean up Spark session
        if spark is not None:
            spark.stop()
            print(f"🔌 Spark session stopped for {table_name}")

