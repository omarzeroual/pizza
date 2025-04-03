import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# load data frame
df = pd.read_csv("data/pizza_final.csv", delimiter = ";", header = 0)

"""
# group by 'city' and count the number of objects
city_counts = df['city'].value_counts()
print(city_counts)

# check for duplicate restaurants
duplicates = df[df.duplicated(subset = "restaurant", keep = False)]
print(duplicates)
# group by restaurant and calculate median price for each
median_prices = df.groupby("restaurant")["price_chf"].median().reset_index()
print(median_prices)
"""

# comparison of prices in different cities
city_avg_price = df.groupby('city')['price_chf'].mean()
# sort cities by prices (descending)
city_avg_price_sorted = city_avg_price.sort_values(ascending=False)
print(city_avg_price_sorted)

# group data by city and extract prices
grouped_prices = [group['price_chf'].values for _, group in df.groupby(['city'])]
# one-way ANOVA
anova_result = stats.f_oneway(*grouped_prices)
print("ANOVA p-value: ", f"{anova_result.pvalue:.6f}")

# post hoc bonferroni
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df['price_chf'], df['city'])
print(tukey) # sig. differences for Basel-Bern, Basel-Biel, Bern-Genf, Biel-Genf,

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

""" Lugano has almost no data points!"""

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







