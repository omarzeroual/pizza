"""
Module Name: cleanertransformer.py
Description: This modules cleans the data of a provided csv file and does certain transformations.
Author:
Date: 2025-03-27
Version: 0.1
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/pizza.csv", delimiter = ";", header = 0)
print("\nDataframe sucessfully loaded.\n")

# Get shape of data frame
nrows = df.shape[0]
ncols = df.shape[1]

print(f"There are {nrows} rows and {ncols} columns in the dataframe.\n")

print(df.info())
print("\n")

# Check for gaps / missing data
# Drop rows where the "restaurant", "location", "product" or "price" column is null or empty
df = df.dropna(subset=["restaurant", "location", "product", "price"])

# Check rows where the "rating", "cuisine", "delivery time", "delivery fee" or "minimum order value" column is null or empty, fill
columns_to_fill = ["rating (number of ratings)", "cuisine", "delivery time", "delivery fee", "minimum order value"]
df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].fillna(np.nan)


# Check if columns show appropriate datatypes, change if needed
# Check if values lie in the expected range
# Identify outliers, treat them reasonably
# Format your dataset suitable for your task (combine, merge, resample, …)
# Enrich your dataset with at least one column of helpful additional information