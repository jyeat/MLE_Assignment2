import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.bronze_layer
import utils.silver_layer
import utils.gold_layer

from pyspark.sql import SparkSession

#Initialize Sparksession
spark = SparkSession.builder \
      .appName("Loan Default EDA") \
      .master("local[*]") \
      .config("spark.driver.host", "127.0.0.1") \
      .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print("Spark session created successfully!")
print(f"Spark version: {spark.version}")

#set up config
snapshot_date_str='2023-01-01'

start_date_str='2023-01-01'
end_date_str='2025-01-01'

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print(dates_str_lst)

#Bronze Table
bronze_lms_directory='datamart/bronze/lms/'

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

for date_str in dates_str_lst:
    utils.bronze_layer.process_bronze_clickstream(date_str,bronze_lms_directory,spark)
    utils.bronze_layer.process_bronze_attributes(date_str,bronze_lms_directory,spark)
    utils.bronze_layer.process_bronze_financials(date_str,bronze_lms_directory,spark)
    utils.bronze_layer.process_bronze_loan(date_str,bronze_lms_directory,spark)

#Silver Table

silver_loan_daily_directory = 'datamart/silver/loan_daily/'
silver_clickstream_directory = 'datamart/silver/clickstream/'
silver_attributes_directory = 'datamart/silver/attributes/'
silver_financials_directory = 'datamart/silver/financials/'

for directory in [silver_loan_daily_directory, silver_clickstream_directory,
                   silver_attributes_directory, silver_financials_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

for date_str in dates_str_lst:
    utils.silver_layer.process_silver_attributes_table(date_str,bronze_lms_directory,silver_attributes_directory,spark)
    utils.silver_layer.process_silver_clickstream_table(date_str,bronze_lms_directory,silver_clickstream_directory,spark)
    utils.silver_layer.process_silver_financials_table(date_str,bronze_lms_directory,silver_financials_directory,spark)
    utils.silver_layer.process_silver_loan_table(date_str,bronze_lms_directory,silver_loan_daily_directory,spark)

#gold table
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory= "datamart/gold/feature_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# run gold backfill
for date_str in dates_str_lst:
    utils.gold_layer.process_feature_gold_table(date_str, silver_attributes_directory,silver_clickstream_directory,silver_financials_directory, gold_feature_store_directory,spark)
    utils.gold_layer.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

