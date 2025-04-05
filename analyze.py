"""
Module Name: analyze.py
Description: This modules is for analysing the pizza price data
Author:
Date: 2025-04-05
Version: 0.1
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
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

# log transform price_chf (due to right skewed distribution)
df['log_price_chf'] = np.log(df['price_chf'])
# comparison of prices in different cities
city_avg_log_price = df.groupby('city')['log_price_chf'].mean()
# group data by city and extract prices
grouped_log_prices = [group['log_price_chf'].values for _, group in df.groupby(['city'])]
# one-way ANOVA comparing log-transformed prices
anova_result = stats.f_oneway(*grouped_log_prices)
print("ANOVA p-value: ", f"{anova_result.pvalue:.6f}")

# post hoc bonferroni
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['log_price_chf'], df['city'])
print(tukey) # sig. differences for Basel-Bern, Basel-Biel, Bern-Genf, Biel-Genf,

"""
# boxplot of prices in different cities
# Set figure size
plt.figure(figsize=(12, 6))
# Create boxplot
sns.boxplot(x="city", y="price_chf", data=df)
# labels
plt.xlabel("City")
plt.ylabel("Pizza Margherita Price (log transformed)")
plt.title("Price Distribution of Pizza Margherita Across Swiss Cities")
# Show plot
plt.show()


# cluster analysis of prices in different cities
# Group by city and get average pizza price
df_cluster = df.groupby("city")["price_chf"].mean().reset_index()
df_cluster = df_cluster[["city", "price_chf"]]
# Normalize the pizza prices
df_cluster["price_scaled"] = (df_cluster["price_chf"] - df_cluster["price_chf"].mean()) / df_cluster["price_chf"].std()
import random

# K-Means initialization (randomly select initial centroids)
k = 3  # You can choose the number of clusters you want
centroids = random.sample(list(df_cluster["price_scaled"]), k)


# Function to assign clusters
def assign_clusters(df, centroids):
    clusters = []
    for price in df["price_scaled"]:
        distances = [abs(price - centroid) for centroid in centroids]
        cluster = distances.index(min(distances))  # Assign to the nearest centroid
        clusters.append(cluster)
    return clusters


# Function to update centroids
def update_centroids(df, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_prices = df[clusters == i]["price_scaled"]
        new_centroids.append(cluster_prices.mean())
    return new_centroids


# Main K-Means algorithm loop
max_iters = 100
for i in range(max_iters):
    # Assign clusters based on current centroids
    df_cluster["Cluster"] = assign_clusters(df_cluster, centroids)

    # Update centroids
    new_centroids = update_centroids(df_cluster, df_cluster["Cluster"], k)

    # If centroids do not change, break the loop
    if new_centroids == centroids:
        break

    centroids = new_centroids

# Check final clusters
print(df_cluster)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot the clustered cities
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_cluster["city"], y=df_cluster["price_chf"], hue=df_cluster["Cluster"], palette="coolwarm", s=100)

plt.xticks(rotation=45)
plt.xlabel("City")
plt.ylabel("Average Pizza Price (CHF)")
plt.title("Cluster Analysis of Pizza Prices in Swiss Cities")
plt.legend(title="Cluster")
plt.show()
"""








