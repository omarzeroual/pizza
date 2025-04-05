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

# Load data
df = pd.read_csv("data/pizza.csv", delimiter=";", header=0)
print(f"{'*' * 60}\nDataFrame successfully loaded.\n{'*' * 60}")

# Show shape
print(f"\nDataFrame shape: {df.shape[0]} rows × {df.shape[1]} columns\n{'*' * 60}")

# Preview structure
print("\nStructure of the DataFrame before cleaning and transforming:\n")
df.info()
print(f"{'*' * 60}")


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

# Set display options
pd.set_option('display.max_rows', 10) # show all rows
pd.set_option('display.max_columns', 10) # show all columns

def generate_summary_statistics(df, variables, digits = 2):
    """
    Generates Summary Statistics incl. median, with specified formatting.

    Parameters:
    - df: pandas DataFrame
    - variables: list or single column name to be included in the summary statistics (default: all columns included)
    - digits: number of decimal places to round to (default 2)

    Returns:
    - Formatted Data Frame with Summary Statistics
    """

    # if a single column name is passed a string, convert it to a list
    if isinstance(variables, str):
        variables = [variables]

    # check that only valid columns are selected
    variables = [col for col in variables if col in df.columns]

    # filter the DataFrame so only the selected columns are included
    df = df[variables]

    stats = df.describe().T
    stats["median"] = df.median(numeric_only=True) # adds median

    # define column order
    cols = ["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]

    summary = stats[cols].round(digits)
    return summary

########################################################################################################################
# Identify outliers, treat them reasonably
########################################################################################################################

def check_outliers(df, variables, lower_quantile=0.025, upper_quantile=0.975):
    """
     Function to check for outliers in the specified variables.

    Parameters:
    - df: pandas DataFrame
    - variables: list of variables (columns to check for outliers)

    Returns:
    - DataFrame with outlier information (counts and bounds)
    """
    limits = df[variables].quantile([lower_quantile, upper_quantile])
    outlier_info = [] # list to store end result

    # loop through the variables to count outliers and store info
    for col in variables:
        lower_bound, upper_bound = limits.loc[lower_quantile, col], limits.loc[upper_quantile, col]
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info.append({
            'Variable': col,
            f'Lower Bound ({lower_quantile*100}%)': lower_bound,
            f'Upper Bound ({upper_quantile * 100}%)': upper_bound,
            'Outlier Count': outliers.shape[0]
        })
    return pd.DataFrame(outlier_info)

def plot_data(df, variables, plot_type = "histogram", bins = 20, rows = 3, cols = 3, figsize = (16,12)):
    """
    Plots different types of plots (histograms, boxplots, dot plots) for a list of variables.

    Parameters:
    - df: pandas DataFrame
    - variables: list of variables (column names) to plot
    - plot_type: type of plot (histogram, boxplot, dotplot)
    - bins: number of bins for histogram (default 20)
    - rows: number of rows for the subplot grid (default 3)
    - cols: number of columns for the subplot grid (default 3)
    - figsize: size of figure (default (16,12))
     """
    # Set up figure with subplots
    plt.figure(figsize=figsize)
    # Loop through the variables to create individual histograms
    for i, var in enumerate(variables, 1):
        plt.subplot(rows, cols, i) # arranges subplots

        if plot_type == "histogram":
            sns.histplot(df[var], kde=True, bins=bins)  # kde=True adds density curve
            plt.title(f'Histogram of {var}')
            plt.xlabel(var)
            plt.ylabel('Frequency')

        if plot_type == "boxplot":
            sns.boxplot(y=df[var])  # specify 'y' for vertical box plot
            plt.title(f'Boxplot of {var}')
            plt.xlabel(var)  # label x-axis

        if plot_type == "dotplot":
            sns.stripplot(x=df[var], jitter=True, alpha=0.5)
            plt.title(f'Dot Plot of {var}')
            plt.xlabel(var)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()

def remove_outliers(df, variables, lower_quantile=0.025, upper_quantile=0.975):
    """
    Removes outliers for the specified variable.

    Parameters:
   -  df: pandas DataFrame
    - variables: column names to remove outliers from (list)
    - lower_quantile: lower bound quantile (default 2.5%)
    -  upper_quantile: upper bound quantile (default 97.5%)

    Returns:
    - df_cleaned: DataFrame with outliers removed
    - outliers: DataFrame with all rows that were removed as outliers
    - removed_outliers_count: count of rows removed as outliers
    """
    if isinstance(variables, str):
        variables = [variables]

    df_cleaned = df.copy()

    for var in variables:
        limits = df[var].quantile([lower_quantile, upper_quantile])
        lower_bound, upper_bound = limits.loc[lower_quantile], limits.loc[upper_quantile]

        outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]
        df_cleaned = df[(df[var] >= lower_bound) & (df[var] <= upper_bound)]

    removed_outliers_count = len(outliers)

    return df_cleaned, outliers, removed_outliers_count

def main():
    # load DataFrame
    df = pd.read_csv("data/pizza_cleaned.csv", delimiter=";", header=0)

    # Variables for outlier analysis
    variables = ['rating', 'num_ratings', 'price_chf', 'delivery_fee_chf', 'min_ord_val_chf',
                 'min_del_time', 'max_del_time']

    # Summary Statistics
    summary_statistics = generate_summary_statistics(df, variables)
    print(f"\nSummary Statistics: \n{summary_statistics}\n{"*" * 60}\n")

    # Plot histograms to get information about the distribution
    plot_data(df, variables, plot_type="histogram", bins=20, rows=3, cols=3)

    # Plot boxplots
    plot_data(df, variables, plot_type="boxplot", rows=3, cols=3)

    # Plot dot plots to check for potential clusters
    plot_data(df, variables, plot_type="dotplot", rows=3, cols=3)

    # Check for outliers
    outliers = check_outliers(df, variables, lower_quantile=0.05, upper_quantile=0.95)
    print("Outlier Information:")
    print(outliers.to_string(index=False))

    # Remove outliers in price
    df_cleaned, removed_outliers, removed_outliers_count = remove_outliers(df, "price_chf", lower_quantile=0.05, upper_quantile=0.95)
    print(f"\nRemoved Outliers in 'price_chf':\n{removed_outliers.to_string(index=False)}")
    print(f"Total number of removed outliers: {removed_outliers_count}")

    df = df_cleaned

    # Plot histogram for 'price' to check distribution after outlier removal
    plot_data(df, ['price_chf'], rows = 1, cols = 1, plot_type="histogram", bins = 20)
    # Plot boxplot for 'price'
    plot_data(df, ['price_chf'], rows = 1, cols = 1, plot_type="boxplot")

    # Summary Statistics
    summary_statistics_after = generate_summary_statistics(df, "price_chf")
    print(f"\nSummary Statistics after Outlier Removal: \n{summary_statistics_after}\n{"*" * 60}\n")
    print("*" * 60 + "\n")

if __name__ == "__main__":
    main()

########################################################################################################################
# Format your dataset suitable for your task (combine, merge, resample, …)
########################################################################################################################

# transform del delivery_fee_chf int categorical variable with two levels
df['delivery_fee_chf_cat'] = np.where(df['delivery_fee_chf'] == 0, 0, 1) # 0 = No Fee, 1 = Fee

# Verify transformation
print("Sample transformed data:")
print(df[['delivery_fee_chf', 'delivery_fee_chf_cat']].head())
print("*"*60+"\n")

# save transformed DataFrame
df.to_csv("data/pizza_transformed.csv", index = False, sep=";")
print("Transformed data frame is saved as a csv file.\n")
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

# Create DataFrame including Regions, Cities and Wages
# Load the wage data
wage_df = pd.read_csv("data/median_wages_2022.csv", delimiter = ";", header = 0)

# Mapping of the regions to the cities
region_to_city_mapping = {
    "Région lémanique": ["Genf", "Lausanne"],
    "Espace Mittelland": ["Bern", "Biel"],
    "Nordwestschweiz": ["Basel"],
    "Zürich": ["Zürich", "Winterthur"],
    "Ostschweiz": ["St. Gallen"],
    "Zentralschweiz": ["Luzern"],
    "Ticino": ["Lugano"]
}

# convert mapping int DataFrame
region_cities_df = pd.DataFrame(
    [(region, city) for region, cities in region_to_city_mapping.items() for city in cities],
    columns=["region", "city"]
)

# standardize region column
region_cities_df['region'] = region_cities_df['region'].str.strip().str.lower()
wage_df['Region'] = wage_df['Region'].str.strip().str.lower()

# merge DataFrames onto 'region' column
wages_cities_df = pd.merge(region_cities_df, wage_df, left_on='region', right_on='Region', how='left')

# drop redundant 'Region' and 'Year' columns
wages_cities_df = wages_cities_df.drop(columns=['Region', 'Year'])

# save wages_city DataFrame to csv
wages_cities_df.to_csv("data/wage_cities.csv", index = False, sep=";")
print("Wage_Cities data frame is saved as a csv file.\n")
print("*"*60+"\n")

# Merge Wages_Cities Data with Enriched Data
# Load enriched data
enriched_df = pd.read_csv("data/pizza_enriched.csv", delimiter = ";", header = 0)

# merge wages_cities_df with pizza_enriched
final_df = pd.merge(enriched_df, wages_cities_df, on = 'city', how = 'left')

# save final df to csv
final_df.to_csv("data/pizza_final.csv", index = False, sep=";")
print("Final data frame is saved as a csv file.\n")
print("*"*60+"\n")




