"""
Module Name: analyze.py
Description: This modules is for analysing the pizza price data
Author:
Date: 2025-04-05
Version: 0.1
"""
import itertools
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from cleanertransformer import generate_summary_statistics as summary_stats

# load DataFrame
df = pd.read_csv("data/pizza_final.csv", delimiter = ";", header = 0)

# summary statistics for prices
summary = summary_stats(df,"price_chf")
print(summary)

# find city with most expensive pizza
max_price = df.groupby("city")["price_chf"].max().sort_values(ascending=False)
print(max_price.round(2))

# find city with cheapest pizza
min_price = df.groupby("city")["price_chf"].min().sort_values(ascending=True)
print(min_price.round(2))

#####
# compare pizza prices between cities
# log transform price_chf (due to right skewed distribution)
df['log_price_chf'] = np.log(df['price_chf'])
# check variance
variance = df.groupby('city')['log_price_chf'].var()
print(f"variance: {variance}")
# group data by city and extract prices
grouped_log_prices = [group['log_price_chf'].values for _, group in df.groupby(['city'])]
# run Kruskal-Wallis test
kruskal_result = kruskal(*grouped_log_prices)
print(f"Kruskal_Wallis Test\n p-value: {kruskal_result.pvalue:.6f}")

# run post-hoc Dunn-Bonferroni test to find which cities differ from each other
dunn_bonf_result = sp.posthoc_dunn(df, val_col='log_price_chf', group_col='city', p_adjust='bonferroni')
alpha = 0.05
sig_results = dunn_bonf_result[dunn_bonf_result < alpha].stack()
sig_results_cleaned = sig_results.reset_index()
print(f"Post Hoc Dunn-Bonferroni, significant results (p-value < 0.05): \n {sig_results_cleaned}")
city_avg_log_price = df.groupby('city')['log_price_chf'].mean().sort_values(ascending=False)
print(city_avg_log_price)

# violin plot of prices in different cities
# set figure size
plt.figure(figsize=(14, 8))
# create boxplot with different colors
city_colours = sns.color_palette("viridis", len(df['city'].unique()))
# dictionary to map cities to colors
city_to_colours = {city: city_colours[i] for i, city in enumerate(df['city'].unique())}
# colour mapping
color_mapping = df['city'].map(city_to_colours)
# create violin plot
ax = sns.violinplot(x="city", y="log_price_chf", data=df,
                    hue='city', palette=city_to_colours,
                    split=False, inner="quart", linewidth=1.5, legend=False)
# overlay individual data points
sns.stripplot(x="city", y="log_price_chf", data=df, color="black", alpha=0.7,
              jitter=True, dodge=True)
# labels and title
plt.xlabel("City", fontsize=14)
plt.ylabel("Pizza Margherita Price (log transformed)", fontsize=14)
plt.title("Price Distribution of Pizza Margherita across Swiss Cities", fontsize=16, weight='bold')
# rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# add gridlines
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# Show plot
plt.tight_layout()
plt.show()

# Test what factors influence a restaurants rating








