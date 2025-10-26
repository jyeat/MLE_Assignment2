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

from pyspark.sql.functions import col, when, sum, abs, size, split, regexp_extract,regexp_replace,expr
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_attributes_table(snapshot_date_str,bronze_lms_directory,silver_attributes_directory,spark):
    
    #Prepare Arguments
    snapshot_date=datetime.strptime(snapshot_date_str,'%Y-%m-%d')

    #connect to bronze
    partition_name="bronze_attributes_daily_"+snapshot_date_str.replace('-','_') +'.csv'
    filepath=bronze_lms_directory + partition_name
    df=spark.read.csv(filepath,header=True,inferSchema=True)
    print('loaded from:',filepath,'rowcount:',df.count())

    # Clean underscore contamination from numeric fields
    numeric_fields_to_clean = ["Age"]

    for field in numeric_fields_to_clean:
      df = df.withColumn(field, regexp_replace(col(field), "_", ""))

    #enforce schema
    column_type_map={
        'Customer_ID':StringType(),
        'Name':StringType(),
        'SSN':StringType(), #SSN is a unique ID
        'Occupation':StringType(),
        'Age':IntegerType(), #Cast cleaned Age string back to integer
        'snapshot_date':DateType()
    }

    for column,new_type in column_type_map.items():
        df=df.withColumn(column,col(column).cast(new_type))

    #handle missingvalue in occupation  
    df=df.withColumn("Occupation",when(col("Occupation").isin("","-","^_+$"),None)
                     .otherwise(col("Occupation")))
    
    #handle invalid age
    df=df.withColumn("Age",when(col("Age").between(16,100),col("Age"))
                     .otherwise(None))
    
    # Mask invalid SSNs
    df = df.withColumn("SSN_is_valid",
                       col("SSN").rlike("^[0-9]{3}-[0-9]{2}-[0-9]{4}$"))
      
    df = df.withColumn("SSN_clean",
                       when(col("SSN_is_valid"), col("SSN"))
                       .otherwise(None))
    
    #augment data
    df=df.withColumn("age_group",
                  when(col('Age')<25,"young")
                  .when(col('Age')<40,"middle_25_40")
                  .when(col('Age')<60,"mature_40_60")
                  .otherwise("seniors"))
    
    df = df.withColumn("occupation_category",
                       when(col("Occupation").isin("Doctor", "Lawyer", "Engineer"),"professional")
                       .when(col("Occupation").isin("Teacher", "Journalist"),"education_media")
                       .when(col("Occupation").isin("Manager", "Entrepreneur"), "business")
                       .when(col("Occupation").isin("Mechanic", "Accountant"), "technical")
                       .when(col("Occupation").isNull(),"None")
                       .otherwise("other"))
    
    #save silver table
    partition_name="silver_attributes_"+snapshot_date_str.replace('-','_')+'.parquet'
    filepath=silver_attributes_directory+ partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:',filepath)

    return df

def process_silver_clickstream_table(snapshot_date_str,bronze_lms_directory,silver_clickstream_directory,spark):

    #prepare arguments
    snapshot_date=datetime.strptime(snapshot_date_str,"%Y-%m-%d")

    #connect to bronze
    partition_name="bronze_clickstream_daily_"+snapshot_date_str.replace('-','_')+ '.csv'
    filepath=bronze_lms_directory+partition_name
    df=spark.read.csv(filepath,header=True,inferSchema=True)
    print('loaded from:',filepath,'row count:',df.count())

    #schema
    column_type_map = {
      "Customer_ID": StringType(),
      "snapshot_date": DateType()}

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Enforce schema for fe_1 through fe_20 columns as there is no 2025-01-01
    for i in range(1, 21):
      df = df.withColumn(f"fe_{i}", expr(f"try_cast(fe_{i} as int)"))

    # numerical aggregation
    # Total activity
    total_expr = col("fe_1")
    for i in range(2, 21):
        total_expr = total_expr + col(f"fe_{i}")
    df = df.withColumn("total_activity", total_expr)

    # Activity intensity
    intensity_expr = abs(col("fe_1"))
    for i in range(2, 21):
        intensity_expr = intensity_expr + abs(col(f"fe_{i}"))
    df = df.withColumn("activity_intensity", intensity_expr)

    # Active features count
    count_expr = when(col("fe_1") != 0, 1).otherwise(0)
    for i in range(2, 21):
        count_expr = count_expr + when(col(f"fe_{i}") != 0, 1).otherwise(0)
    df = df.withColumn("active_features_count", count_expr)

    #now to silver table
    partition_name= 'silver_clickstream_'+ snapshot_date_str.replace('-','_')
    filepath= silver_clickstream_directory + partition_name +'.parquet'
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:',filepath)

    return df 

def process_silver_financials_table(snapshot_date_str, bronze_lms_directory, silver_financials_directory,spark):

    #prepare arguments
    snapshot_date=datetime.strptime(snapshot_date_str,"%Y-%m-%d")

    #connect to bronze table
    partition_name="bronze_financials_daily_"+snapshot_date_str.replace('-','_')+'.csv'
    filepath=bronze_lms_directory+partition_name
    df=spark.read.csv(filepath,header=True,inferSchema=True)
    print('loaded from:', filepath)

    # Clean underscore contamination from numeric fields
    numeric_fields_to_clean = ["Annual_Income","Monthly_Inhand_Salary",
                               "Num_Bank_Accounts","Num_Credit_Card",
                               "Interest_Rate","Num_of_Loan",
                               "Delay_from_due_date","Num_of_Delayed_Payment",
                               "Changed_Credit_Limit",
                                "Outstanding_Debt", "Credit_Utilization_Ratio",
                                "Total_EMI_per_month","Amount_invested_monthly",
                                "Monthly_Balance"]

    for field in numeric_fields_to_clean:
      df = df.withColumn(field, regexp_replace(col(field), "_", ""))

    #schemacast
    column_type_map={
    'Customer_ID':StringType(),
    'Annual_Income':FloatType(),
    'Monthly_Inhand_Salary':FloatType(),
    # 'Num_Bank_Accounts':IntegerType(),
    # 'Num_Credit_Card':IntegerType(),
    'Interest_Rate':FloatType(),
    # 'Num_of_Loan':IntegerType(),
    'Type_of_Loan':StringType(), #categorical
    # 'Delay_from_due_date':IntegerType(),
    # 'Num_of_Delayed_Payment':IntegerType(),
    'Changed_Credit_Limit':FloatType(),
    # 'Num_Credit_Inquiries':IntegerType(),
    'Credit_Mix':StringType(),
    'Outstanding_Debt':FloatType(),
    'Credit_Utilization_Ratio':FloatType(),
    'Credit_History_Age':StringType(),  #keep as string first
    'Payment_of_Min_Amount':StringType(),
    'Total_EMI_per_month':FloatType(),
    'Amount_invested_monthly':FloatType(),
    'Payment_Behaviour':StringType(), #categorical
    'Monthly_Balance':FloatType(),
    'snapshot_date':DateType()}

    for column, new_type in column_type_map.items():
        df=df.withColumn(column, expr(f"try_cast({column} as {new_type.simpleString()})"))


    #trycast for integertype
    integer_columns = ["Num_Bank_Accounts", "Num_Credit_Card", "Num_of_Loan",
                    "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]

    for col_name in integer_columns:
        df = df.withColumn(col_name, expr(f"try_cast({col_name} as int)"))

    #missing value for stringtype
    df=df.withColumn("Credit_Mix",
                     when(col("Credit_Mix").isin("_",""),None)
                     .otherwise(col("Credit_Mix")))
    
    df = df.withColumn("Type_of_Loan",
                       when(col("Type_of_Loan").isin("", "_", "Not Specified"), None)
                       .otherwise(col("Type_of_Loan")))

    #clean categorical column
    valid_behaviors = [
      "High_spent_Large_value_payments",
      "High_spent_Medium_value_payments",
      "High_spent_Small_value_payments",
      "Low_spent_Large_value_payments",
      "Low_spent_Medium_value_payments",
      "Low_spent_Small_value_payments"]
    df = df.withColumn("Payment_Behaviour",
                       when(col("Payment_Behaviour").isin(valid_behaviors),col("Payment_Behaviour"))
                       .otherwise(None))

    #clean negative integer/float
    positive_column=['Annual_Income',
                     'Monthly_Inhand_Salary',
                     'Num_Bank_Accounts',
                     'Num_Credit_Card',
                     'Num_of_Loan',
                     'Num_of_Delayed_Payment',
                     'Num_Credit_Inquiries',
                     'Outstanding_Debt',
                     'Credit_Utilization_Ratio',
                     'Total_EMI_per_month',
                     'Amount_invested_monthly']
    
    for c in positive_column:
        df=df.withColumn(c,when(col(c)>=0,col(c))
                         .otherwise(None))
        
    #handle extreme outlier found out in eda
    outlier_bounds = {
        "Num_Bank_Accounts": (0, 10),
        "Num_Credit_Card": (0, 15),
        "Interest_Rate": (0, 50),
        "Num_Credit_Inquiries": (0, 50),
        "Total_EMI_per_month": (0, 10000),
        "Num_of_Loan": (0, 20),
        "Num_of_Delayed_Payment": (0, 100),
        "Annual_Income": (0, 1000000),
        "Amount_invested_monthly": (0, 10000),
        "Monthly_Balance": (-10000, 100000),
        "Outstanding_Debt": (0, 50000),
        "Delay_from_due_date": (0, 365)
    }

    for col_name, (min_val, max_val) in outlier_bounds.items():
        df = df.withColumn(col_name,
                            when(col(col_name).between(min_val, max_val), col(col_name))
                            .otherwise(None))
    
    #augmented data

    #credit month total
    df=df.withColumn("Credit_history_total_months",
                     when(col("Credit_History_Age").isNull(),None)
                     .otherwise(
                         regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast("int") * 12 +
                         regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast("int"))
                         )
    
    #financial health
    df=df.withColumn("Debt_to_income_ratio",
                     when(col("Annual_Income")>0,
                          col("Outstanding_Debt")/col("Annual_Income"))
                          .otherwise(None))
    
    #credit utilization risk
    df = df.withColumn("credit_risk_level",
                       when(col("Credit_Utilization_Ratio") > 90, "critical")
                       .when(col("Credit_Utilization_Ratio") > 70, "very_high")
                       .when(col("Credit_Utilization_Ratio") > 50, "high")
                       .when(col("Credit_Utilization_Ratio") > 30, "moderate")
                       .when(col("Credit_Utilization_Ratio") > 10, "low")
                       .otherwise("very_low"))
    
    
    #now to silver table
    partition_name= 'silver_financials_'+ snapshot_date_str.replace('-','_') 
    filepath= silver_financials_directory + partition_name +'.parquet'
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:',filepath)

    return df 

def process_silver_loan_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(expr("try_divide(overdue_amt, due_amt)")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_silver_table_main(table_name, silver_db, bronze_db, snapshot_date_str):
    """
    Main function to process any silver table
    Called by Airflow DAG with op_kwargs
    
    Args:
        table_name: 'clickstream', 'attributes', 'financials', or 'lms'
        silver_db: Path to silver database directory
        bronze_db: Path to bronze database directory
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
    """
    import pyspark
    
    # Create Spark session
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver-pipeline") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Make sure directories end with '/'
    if not bronze_db.endswith('/'):
        bronze_db = bronze_db + '/'
    if not silver_db.endswith('/'):
        silver_db = silver_db + '/'
    
    # Route to correct function based on table_name
    if table_name == 'clickstream':
        process_silver_clickstream_table(snapshot_date_str, bronze_db, silver_db + 'clickstream/', spark)
    elif table_name == 'attributes':
        process_silver_attributes_table(snapshot_date_str, bronze_db, silver_db + 'attributes/', spark)
    elif table_name == 'financials':
        process_silver_financials_table(snapshot_date_str, bronze_db, silver_db + 'financials/', spark)
    elif table_name == 'lms':
        process_silver_loan_table(snapshot_date_str, bronze_db, silver_db + 'loan_daily/', spark)
    else:
        raise ValueError(f"Unknown table_name: {table_name}")
    
    print(f"✅ Silver {table_name} completed for {snapshot_date_str}")