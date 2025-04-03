"""
Module Name: cleanertransformer.py
Description: This modules cleans the data of a provided csv file and does certain transformations.
Author:
Date: 2025-04-01
Version: 0.3
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
print("*"*60+"\n")
print("Dataframe sucessfully loaded.\n")
print("*"*60+"\n")

# Get shape of data frame
nrows = df.shape[0]
ncols = df.shape[1]

print(f"There are {nrows} rows and {ncols} columns in the dataframe.\n")
print("*"*60+"\n")

print("Structure of the data frame before cleaning and transforming:\n")
print(df.info())
print("*"*60+"\n")

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
# Convert plz to integer
df['plz'] = df['plz'].astype(int)

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

# Drop unnecessary columns
df.drop(columns=["location", "rating (number of ratings)" ,"price", "delivery time", "delivery fee", "minimum order value"], inplace=True)

print("Structure of the data frame after cleaning and transforming:\n")
print(df.info())
print("*"*60+"\n")

# save cleaned df to csv
df.to_csv("data/pizza_cleaned.csv", index = False, sep=";")
print("Cleaned data frame is saved as a csv file.\n")
print("*"*60+"\n")

########################################################################################################################
# Check if values lie in the expected range
########################################################################################################################

# Summary Statistics
pd.set_option('display.max_rows', 10) # show all rows
pd.set_option('display.max_columns', 10) # show all columns

summary_statistics = df.describe().T
# add median to summary statistics
summary_statistics["median"] = df.median(numeric_only=True)
# reorder columns
column_order = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
summary_statistics = summary_statistics[column_order]
# exclude first variable "plz" and round digits up to 2 decimal places
summary_statistics = summary_statistics.iloc[1: ].round(2)

print(f"Summary Statistics: \n {summary_statistics} \n")
print("*"*60+"\n")

########################################################################################################################
# Identify outliers, treat them reasonably
########################################################################################################################
# list of numeric variables for outlier analysis
variables = ['rating', 'num_ratings', 'price_chf', 'delivery_fee_chf', 'min_ord_val_chf',
             'min_del_time', 'max_del_time']

# Visualize distribution with histograms
# Set up figure with subplots
plt.figure(figsize=(16, 12))
# Loop through the variables to create individual histograms
for i, var in enumerate(variables, 1):
    plt.subplot(3, 3, i)  # arrange subplots in a 3x3 grid
    sns.histplot(df[var], kde=True, bins=20)  # kde=True adds density curve
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()

# Visualize outliers with boxplots
# Set up the figure with subplots
plt.figure(figsize=(8, 12))
# Loop through the variables to create individual boxplots
for i, variable in enumerate(variables, 1):
    plt.subplot(4, 2, i)  # arrange subplots in a 7x1 grid
    sns.boxplot(y=df[variable])  # specify 'y' for vertical box plot
    plt.title(f'Boxplot of {variable}')
    plt.xlabel(variable) # label x-axis
# Adjust layout to prevent overlap
plt.tight_layout()
# Increase vertical space between plots
plt.subplots_adjust(hspace=0.5)
# Show plot
plt.show()

limits = df[variables].quantile([0.025, 0.975])
# dictionnary to store outliers count for each variable
outlier_count = {}
# list to store end result
outlier_info = []
# loop through the variables to count outliers and store info
for col in variables:
    lower_bound = limits.loc[0.025, col]
    upper_bound = limits.loc[0.975, col]
    # count number of outliers outside the 2.5% to 97.5% range
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count[col] = outliers.shape[0]  # store outlier count
    # add to result list
    outlier_info.append({
        'Variable': col,
        'Lower Bound (2.5%)': lower_bound,
        'Upper Bound (97.5%)': upper_bound,
        'Outlier Count': outlier_count[col]
    })
# convert the list outlier_info into a DataFrame
outlier_info_df = pd.DataFrame(outlier_info)
# display the data frame with the limits and outlier counts
print("Outlier Info:")
print(outlier_info_df.to_string(index=False))
print("*"*60+"\n")

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

### Remove outliers in price
prc_lwr_bnd = limits.loc[0.025, "price_chf"]
prc_upr_bnd = limits.loc[0.975, "price_chf"]

# Remove outliers from price column
df = df[(df["price_chf"] >= prc_lwr_bnd) & (df["price_chf"] <= prc_upr_bnd)]

sns.histplot(df["price_chf"], kde=True, bins=20)  # kde=True adds a density curve
plt.title(f'Histogram of price_chf')
plt.xlabel("price_chf")
plt.ylabel('Frequency')
plt.show()

summary_statistics_after = df.describe().T
# add median to summary statistics
summary_statistics_after["median"] = df.median(numeric_only=True)
# reorder columns
column_order = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
summary_statistics_after = summary_statistics_after[column_order]
# exclude first variable "plz" and round digits up to 2 decimal places
summary_statistics = summary_statistics_after.iloc[1: ].round(2)

print(f"Summary Statistics after Outlier Elimination: \n {summary_statistics} \n")
print("*"*60+"\n")
########################################################################################################################
# Format your dataset suitable for your task (combine, merge, resample, …)
########################################################################################################################

# transform del delivery_fee_chf int categorical variable with two levels
df['delivery_fee_chf_cat'] = np.where(df['delivery_fee_chf'] == 0, 0, 1) # 0 = No Fee, 1 = Fee

# Verify transformation
print("Sample transformed data:")
print(df[['delivery_fee_chf', 'delivery_fee_chf_cat']].head())
print("*"*60+"\n")

# save transformed df
df.to_csv("data/pizza_transformed.csv", index = False, sep=";")
print("Log transformed data frame is saved as a csv file.\n")
print("*"*60+"\n")

########################################################################################################################
# Enrich your dataset with at least one column of helpful additional information
########################################################################################################################

# Add geographical information
# Read data set with geographical information for each city/town in Switzerland
df_coordinates = pd.read_csv("data/AMTOVZ_CSV_WGS84.csv", delimiter = ";", header = 0)

# Filter for necessary columns
df_coord_slct = df_coordinates[["PLZ", "E", "N"]]

# Merge data frames on plz column
merged_df = pd.merge(df, df_coord_slct, left_on="plz", right_on="PLZ", how='inner')

# Drop the redundant 'PLZ' column from merge
merged_df = merged_df.drop(columns=['PLZ'])

# Rename new columns for clarity
merged_df = merged_df.rename(columns={"E": "city_E", "N": "city_N"})

# save enriched df to csv
merged_df.to_csv("data/pizza_enriched.csv", index = False, sep=";")
print("Enriched data frame is saved as a csv file.\n")
print("*"*60+"\n")

# Merge with Wages
# Load the wage data
wage_df = pd.read_csv("data/median_wages_2022.csv", delimiter = ";", header = 0)

# Mapping for the region to city
region_to_city_mapping = {
    "Région lémanique": ["Genf", "Lausanne"],
    "Espace Mittelland": ["Bern", "Biel"],
    "Nordwestschweiz": ["Basel"],
    "Zürich": ["Zürich", "Winterthur"],
    "Ostschweiz": ["St. Gallen"],
    "Zentralschweiz": ["Luzern"],
    "Ticino": ["Lugano"]
}

# convert mapping to data frame
region_cities_df = pd.DataFrame(
    [(region, city) for region, cities in region_to_city_mapping.items() for city in cities],
    columns=["region", "city"]
)

# standardize region column
region_cities_df['region'] = region_cities_df['region'].str.strip().str.lower()
wage_df['Region'] = wage_df['Region'].str.strip().str.lower()

# merge data frames onto 'region' column
wages_cities_df = pd.merge(region_cities_df, wage_df, left_on='region', right_on='Region', how='left')

# drop redundant 'Region' and 'Year' columns
wages_cities_df = wages_cities_df.drop(columns=['Region', 'Year'])

# save wages_city data frame to csv
wages_cities_df.to_csv("data/wage_cities.csv", index = False, sep=";")
print("Wage_Cities data frame is saved as a csv file.\n")
print("*"*60+"\n")

# Load enriched data
enriched_df = pd.read_csv("data/pizza_enriched.csv", delimiter = ";", header = 0)

# merge wages_cities_df with pizza_enriched
final_df = pd.merge(enriched_df, wages_cities_df, on = 'city', how = 'left')

# save final df to csv
final_df.to_csv("data/pizza_final.csv", index = False, sep=";")
print("Final data frame is saved as a csv file.\n")
print("*"*60+"\n")




