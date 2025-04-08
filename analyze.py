<<<<<<< visualizations_vl
"""
Module Name: analyze.py
Description: This modules is for analysing the pizza price data
Author:
Date: 2025-04-05
Version: 0.1
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import kruskal
from cleanertransformer import generate_summary_statistics as summary_stats

# load DataFrame
df = pd.read_csv("data/pizza_final.csv", delimiter = ";", header = 0)

# summary statistics for prices
summary = summary_stats(df,"price_chf")
print(summary)

# find city with most expensive pizza
city_prices = df.groupby("city")["price_chf"].agg(["max", "min"]).round(2)
print(city_prices["max"].sort_values(ascending=False))
print(city_prices["min"].sort_values(ascending=True))

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
sig_results = dunn_bonf_result[dunn_bonf_result < alpha].stack().reset_index()
print(f"Post Hoc Dunn-Bonferroni, significant results (p-value < 0.05): \n {sig_results}")
# average log prices per city
city_avg_log_price = df.groupby('city')['log_price_chf'].mean().sort_values(ascending=False)
print(city_avg_log_price)

# Violin plot for prices in different cities
plt.figure(figsize=(14, 8))
sns.violinplot(x="city", y="log_price_chf", data=df, hue='city', split=False, inner="quart", linewidth=1.5, legend=False, palette="viridis")
# make individual data points visible
sns.stripplot(x="city", y="log_price_chf", data=df, color="black", alpha=0.7, jitter=True, dodge=True)
# labels and title
plt.xlabel("City", fontsize=14)
plt.ylabel("Pizza Margherita Price (log transformed)", fontsize=14)
plt.title("Price Distribution of Pizza Margherita across Swiss Cities", fontsize=16, weight='bold')
# rotation of x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
# add gridlines for orientation
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
# Show plot
plt.tight_layout() # avoids overlap
plt.show()






