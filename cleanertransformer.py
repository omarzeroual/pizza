"""
Module Name: cleanertransformer.py
Description: This modules cleans the data of a provided csv file and does certain transformations.
Author:
Date: 2025-04-01
Version: 0.2
"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

def extract_rating_info(rating_str):
    """Function to extract rating and number of ratings using regex"""
    if isinstance(rating_str, str):
        match = re.match(r"(\d+|\d+,\d+)\((\d+)\+?\)", rating_str)
        if match:
            rating = float(match.group(1).replace(',', '.'))
            num_ratings = int(match.group(2))
            return rating, num_ratings
    return 0, 0

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
        else:
            # in case of only a single delivery time (e.g. "20")
            return float(del_time_str), float(del_time_str)
    return 0, 0
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

# save cleaned df to csv
df.to_csv("data/pizza_cleaned.csv", index = False, sep=";")
########################################################################################################################
# Check if values lie in the expected range
########################################################################################################################

# Summary Statistics
pd.set_option('display.max_rows', 100) # show all 7 rows
pd.set_option('display.max_columns', 100) # show all 8 columns

print("Summary Statistics:\n")
summary_statistics = df.describe().T
# add median to summary statistics
summary_statistics["median"] = df.median(numeric_only=True)
# reorder columns
column_order = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
summary_statistics = summary_statistics[column_order]
print(summary_statistics)

""" potential outliers in price_chf min. and max. price 4.5 to 65.0 seem unlikely.
max for min_ord_val_chf seems very high with 150 CHF.
big range for num_ratings is plausible -> might reflect popularity of restaurant.
other ranges look reasonable """

########################################################################################################################
# Identify outliers, treat them reasonably
########################################################################################################################

# Plot histograms to visually check distribution
# List of the variables to plot
variables = ['rating', 'num_ratings', 'price_chf', 'delivery_fee_chf', 'min_ord_val_chf', 'min_del_time', 'max_del_time']

# Set up the matplotlib figure with subplots
plt.figure(figsize=(16, 12))

# Loop through the variables to create individual histograms
for i, var in enumerate(variables, 1):
    plt.subplot(3, 3, i)  # Arrange subplots in a 3x3 grid (adjust as necessary)
    sns.histplot(df[var], kde=True, bins=20)  # kde=True adds a density curve
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

""" no normal distribution for any variable """

# Set up the figure for the box plots with adjusted figure size
plt.figure(figsize=(8, 12))

# Plot boxplots for each variable
for i, variable in enumerate(variables, 1):
    plt.subplot(4, 2, i)  # Arrange subplots in a 7x1 grid
    sns.boxplot(y=df[variable])  # Specify 'y' for vertical box plot (variable is plotted on y-axis)
    plt.title(f'Boxplot of {variable}')

    # Label x-axis with the variable name for clarity
    plt.xlabel(variable)

# Adjust layout to make figures bigger
plt.tight_layout()
plt.subplots_adjust(hspace=0.5) # Increase vertical space between plots

# Show plot
plt.show()

# percentile method - define limits
lower_bound = 0.025
upper_bound = 0.975

# compute lower and upper limit for outliers
outlier_bounds = df[variables].quantile([lower_bound, upper_bound])

# Iterate through each column and display outliers
for col in variables:
    try:
        lower_bound = outlier_bounds.loc[lower_bound, col]
        upper_bound = outlier_bounds.loc[upper_bound, col]
    except KeyError:
        lower_bound = outlier_bounds.iloc[0][col]  # Fallback if KeyError
        upper_bound = outlier_bounds.iloc[1][col]

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"\n Outliers in '{col}':")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    print(outliers[[col]])  # Show only the outlier column for clarity

# log transformation
log_transform_cols = ['price_chf', 'delivery_fee_chf', 'min_ord_val_chf', 'min_del_time', 'max_del_time']

# Apply log transformation
for col in log_transform_cols:
    df[f'log_{col}'] = np.log1p(df[col])  # log1p(x) = log(x + 1)

# Display transformed columns
df[[f'log_{col}' for col in log_transform_cols]].head()

# plot histograms
fig, axes = plt.subplots(nrows=2, ncols=len(log_transform_cols), figsize=(15, 6))

for i, col in enumerate(log_transform_cols):
    # Before transformation
    sns.histplot(df[col], bins=30, kde=True, ax=axes[0, i])
    axes[0, i].set_title(f'Original {col}')

    # After transformation
    sns.histplot(df[f'log_{col}'], bins=30, kde=True, ax=axes[1, i])
    axes[1, i].set_title(f'Log Transformed {col}')

plt.tight_layout()
plt.show()

# dot plot
# variables to plot
variables = ['price_chf', 'delivery_fee_chf', 'min_ord_val_chf', 'min_del_time', 'max_del_time']

# figure size
plt.figure(figsize=(10, len(variables) * 2))

# create dot plots for each variable
for i, var in enumerate(variables, 1):
    plt.subplot(len(variables), 1, i)  # Create a subplot for each variable
    sns.stripplot(x=df[var], jitter=True, alpha=0.6)  # Dot plot (strip plot)
    plt.title(f"Dot Plot of {var}")
    plt.xlabel(var)

# Adjust layout for better readability
plt.tight_layout()
plt.show()

########################################################################################################################
# Format your dataset suitable for your task (combine, merge, resample, …)
########################################################################################################################

########################################################################################################################
# Enrich your dataset with at least one column of helpful additional information
########################################################################################################################
