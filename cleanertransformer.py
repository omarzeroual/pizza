"""
Module Name: cleanertransformer.py
Description: This modules cleans the data of a provided csv file and does certain transformations.
Author:
Date: 2025-03-27
Version: 0.1
"""
import pandas as pd
import numpy as np
import re

def extract_rating_info(rating_str):
    """Function to extract rating and number of ratings using regex"""
    if isinstance(rating_str, str):
        match = re.match(r"(\d+|\d+,\d+)\((\d+)\+?\)", rating_str)
        if match:
            rating = float(match.group(1).replace(',', '.'))
            num_ratings = int(match.group(2))
            return rating, num_ratings
    return np.nan, 0 # Set 0 for no ratings

def extract_price(price_str):
    """Function to extract price using regex"""
    match = re.search(r"(\d+,\d+)", price_str)
    if match:
        price = float(match.group(1).replace(',', '.'))
        return price
    else:
        return np.nan

def extract_delfee(delfee_str):
    """Function to extract delivery fee using regex"""
    match = re.match(r"(\d+,\d+|\d+)|(Gratis.*|Kostenlos.*)", delfee_str)
    if match:
        if match.group(1):
            fee = float(match.group(1).replace(',', '.'))
        else:
            fee = 0 # Set the fee to zero if "Gratis" or "Kostenlos*" is mentioned in the fee string
        return fee
    else:
        return np.nan

def extract_del_time(del_time_str):
    """Function to extract delivery time using regex"""
    if isinstance(del_time_str, str):
        match = re.search(r"(\d+)-(\d+)", del_time_str)
        if match:
            min_del_time = float(match.group(1))
            max_del_time = float(match.group(2))
            return min_del_time, max_del_time
    return np.nan, np.nan

########################################################################################################################
# Load data
########################################################################################################################
df = pd.read_csv("data/pizza.csv", delimiter = ";", header = 0)
print("\nDataframe sucessfully loaded.\n")

# Get shape of data frame
nrows = df.shape[0]
ncols = df.shape[1]

print(f"There are {nrows} rows and {ncols} columns in the dataframe.\n")

print(df.info())
print("\n")

########################################################################################################################
# Check for gaps / missing data
########################################################################################################################

# Drop rows where the "restaurant", "location", "product" or "price" column is null or empty
df = df.dropna(subset=["restaurant", "location", "product", "price"])

# Check rows where the "rating", "cuisine", "delivery time", "delivery fee" or "minimum order value" column is null or empty, fill
columns_to_fill = ["rating (number of ratings)", "cuisine", "delivery time", "delivery fee", "minimum order value"]
df.loc[:, columns_to_fill] = df.loc[:, columns_to_fill].fillna(np.nan)

########################################################################################################################
# Check if columns show appropriate datatypes, change if needed
########################################################################################################################

# Split the location into new columns for city and postcode
df[["plz", "city"]] = df["location"].str.split(" ", n = 1, expand=True)

# Split the values in the 'rating (number of ratings)' column with function and create new columns
df[['rating', 'num_ratings']] = df['rating (number of ratings)'].apply(lambda x: pd.Series(extract_rating_info(x)))
df["num_ratings"] = df["num_ratings"].astype(int)

# Extract the float from the price column and create a new column
df["price_chf"] = df["price"].apply(lambda x: pd.Series(extract_price(x)))

# Extract the float from the delivery fee column and create new column
df["delivery_fee_chf"] = df["delivery fee"].apply(lambda x: pd.Series(extract_delfee(x)))

# Extract the float from the minimum order value column and create new column
df["min_ord_val_chf"] = df["minimum order value"].apply(lambda x: pd.Series(extract_price(x)))

# Split the values in the "delivery time" column with function and create new columns
df[["min_del_time", "max_del_time"]] = df["delivery time"].apply(lambda x: pd.Series(extract_del_time(x)))

print(df.info())

########################################################################################################################
# Check if values lie in the expected range
########################################################################################################################



########################################################################################################################
# Identify outliers, treat them reasonably
########################################################################################################################

########################################################################################################################
# Format your dataset suitable for your task (combine, merge, resample, …)
########################################################################################################################

########################################################################################################################
# Enrich your dataset with at least one column of helpful additional information
########################################################################################################################
