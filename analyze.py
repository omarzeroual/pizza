"""
Module Name: analyze.py
Description: This modules is for analysing the pizza price data
Author:
Date: 2025-04-05
Version: 0.1
"""
from cleanertransformer import generate_summary_statistics as summary_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import statsmodels.api as sm
from scipy.stats import kruskal

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
print(f"variance: {variance.round(2)}")
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
print(city_avg_log_price.round(2))

# Violin plot for prices in different cities
plt.figure(figsize=(14, 8))
sns.violinplot(x="city", y="log_price_chf", data=df, hue='city', split=False, inner="box", linewidth=1.5, legend=False, palette="viridis")
# make individual data points visible
# sns.stripplot(x="city", y="log_price_chf", data=df, color="black", alpha=0.7, jitter=True, dodge=True)
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


##### Test what factors influence a restaurants rating
# clean/group column cuisine
num_rows = df.shape[0] # check number of rows before transformation
print(num_rows)
unique_cuisines = df['cuisine'].unique()
print(unique_cuisines)
cuisine_groups = {
    'Italian': ['Italienisch', 'Italienische Pizza', 'Pasta', 'Pizza'],
    'MiddleEastern': ['Türkisch', 'Döner', 'Kebab', 'Türkische Pizza', 'Pide', 'Balkanküche',
                'Arabisch', 'Falafel', 'Meze', 'Libanesisch'],
    'Indian' : ['Indisch'],
    'Swiss': ['Schweizer Küche'],
    'Fastfood': ['Snacks', 'Sandwiches', 'Wraps', 'Amerikanisch', 'Burger', 'Hot Dog'],
    'Other': ['Lebensmittel', 'Alkohol', 'Andere Getränke', 'Lokale Geheimtipps', 'Mittagsangebote', '-25% Deals',
              'Gourmet']
}

def group_cuisine(cuisine):
    for group, cuisines in cuisine_groups.items():
        if any(cuisine_part in cuisine for cuisine_part in cuisines):
            return group
    return 'Other'

df['cuisine_group'] = df['cuisine'].apply(group_cuisine)
print(df['cuisine_group'].unique())
print(df['cuisine_group'].value_counts())

# log transform price and min. order value
df['log_price_chf'] = np.log(df['price_chf'])
df['log_min_ord_val_chf'] = np.log(df['min_ord_val_chf'])
# transform num ratings in to categorical variable
df['num_ratings_cat'] = pd.qcut(df['num_ratings'], q=3, labels=['Low', 'Medium', 'High'])
# Convert categorical variables into dummies
df_dummies = pd.get_dummies(df[['num_ratings_cat', 'cuisine_group', 'delivery_fee_chf_cat', 'city', 'region']], drop_first=True)
df_dummies = df_dummies.astype(int)
# Combine dummies and numeric variables
var_num = ['log_price_chf', 'log_min_ord_val_chf', 'Median Monthly Wage (CHF)']
X = pd.concat([df[var_num], df_dummies], axis=1)
print(X.dtypes)
y = df['rating']
# Add constant (intercept)
X_const = sm.add_constant(X)
# Fit OLS model
ols_model = sm.OLS(y, X_const).fit()
# print output
print(ols_model.summary())






