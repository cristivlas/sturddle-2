#! /usr/bin/env python3
'''
Use statistical method: interquartile range (IQR) to identify and remove outlier data points.
'''
import argparse
import sqlite3

import numpy as np
import pandas as pd
import tqdm
import ydata_profiling

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_db', help='Input SQLite database file')
parser.add_argument('output_db', help='Output SQLite database file')
parser.add_argument('--outlier-threshold', type=float, default=1.5, help='Threshold for outlier detection (in terms of IQR)')
parser.add_argument('--report-file', default='report.html', help='File to save the cleaning report')
args = parser.parse_args()

# Connect to the input SQLite database
conn = sqlite3.connect(args.input_db)

# Load the data into a pandas DataFrame
df = pd.read_sql_query('SELECT * FROM position', conn)

# Define the evaluation score column name
score_col = 'score'

# Compute the interquartile range (IQR)
q1 = df[score_col].quantile(0.25)
q3 = df[score_col].quantile(0.75)
iqr = q3 - q1

# Define the threshold for outliers (in terms of IQR)
outlier_thresh = args.outlier_threshold * iqr

# Identify the outlier data points
outliers = df[np.abs(df[score_col] - df[score_col].median()) > outlier_thresh]

# Remove the outlier data points
# df = df[~df.index.isin(outliers.index)]
# Remove the outlier data points with progress
with tqdm.tqdm(total=len(df)) as pbar:
    for idx in outliers.index:
        df = df.drop(idx)
        pbar.update(1)

# Save the cleaned data to a new SQLite database
clean_conn = sqlite3.connect(args.output_db)
df.to_sql('position', clean_conn, index=False, if_exists='replace')

# Generate a report of the cleaning process
profile = ydata_profiling.ProfileReport(df, minimal=True)
profile.to_file(args.report_file)

# Close the database connections
conn.close()
clean_conn.close()
