"""
Module Name: analyze.py
Description: This modules is for analysing the pizza price data
Author: Valérie
Date: 2025-04-09
Version: 1.0
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import statsmodels.api as sm
from scipy.stats import kruskal
from cleanertransformer import generate_summary_statistics as summary_stats

def load_data(filepath, delimiter=";", header=0):
    """ Loads the dataset from a CSV file

    Parameters:
    - filepath: path to the CSV file
    - delimiter: delimiter of the CSV file (default ";")
    - header: the row number to use as header (default 0)

    Returns:
    - pd.DataFrame: pandas DataFrame with the loaded data
     """
    return pd.read_csv(filepath, delimiter=delimiter, header=header)

def log_transform(df, vars):
    """ Log transforms the specified variables

    Parameters:
    - df: pandas DataFrame
    - vars: list of variables to log transform

    Returns:
    - pd.DataFrame: DataFrame with the log transformed variables
    """
    for var in vars:
        df[f"log_{var}"] = np.log(df[var])
    return df

def kruskal_wallis(df, var, group_var):
    """
    Runs a Kruskal Wallis Test to test for differences between groups in the specified variable.
    Non-parametric test, applicable when the assumptions of ANOVA like normal distribution are not met.

    Parameters:
    - df: pandas DataFrame
    - var: dependent numeric variable (str)
    - group_var: independent categorical variable to group by (str)

    Returns:
    - kruskal-wallis result: test statistics including p-values from the Kruskal Wallis test
    """
    groups = [group[var].values for _, group in df.groupby(group_var)]
    return kruskal(*groups)

def dunn_posthoc(df, var, group_var, p_adjust='bonferroni'):
    """ Runs a Dunn's Posthoc Test on the specified variable grouped by a categorical variable
    to determine which groups differ from each other.

    Parameters:
    - df: pandas DataFrame
    - var: dependent numeric variable (str)
    - group_var: independent categorical variable to group by (str)
    - p_adjust: method to adjust p-values for multiple comparisons (default: 'bonferroni')

    Returns:
    - dunn's posthoc result: p-values for pairwise group comparisons
    """
    return sp.posthoc_dunn(df, var, group_var, p_adjust)

def fit_ols_model(df, target, num_vars, cat_vars):
    """
    Fit an OlS regression model to predict the target variable based on the specified numeric and
    categorical variables.

    Parameters:
    - df: pandas DataFrame
    - target: dependent variable (str)
    - num_vars: numerical predictor variables (list)
    - cat_vars: categorical predictor variables (list)

    Returns:
    - ols_model: fitted OLS regression model
    """
    # Convert categorical variables into dummies
    df_dummies = pd.get_dummies(df[cat_vars], drop_first=True)
    df_dummies = df_dummies.astype(int)
    # Combine dummies and numeric variables
    X = pd.concat([df[num_vars], df_dummies], axis=1)
    # Set target variable
    y = df[target]
    # Add constant (intercept)
    X_const = sm.add_constant(X)
    # Fit OLS model
    ols_model = sm.OLS(y, X_const).fit()

    return ols_model

def main():
    ########################################################################################################################
    # Load Data
    ########################################################################################################################

    df = load_data("data/pizza_final.csv", delimiter = ";", header = 0)

    ########################################################################################################################
    # Summary Statistics incl. min, max pizza prices
    ########################################################################################################################
    # summary statistics for prices
    print(summary_stats(df,"price_chf"))

    # find city with most expensive and cheapest pizza
    city_prices = df.groupby("city")["price_chf"].agg(["max", "min"]).round(2)
    print(city_prices["max"].sort_values(ascending=False))
    print(city_prices["min"].sort_values(ascending=True))

    ########################################################################################################################
    # Data Preparation for further Analysis
    ########################################################################################################################
    # log transform price_chf and min_ord_val_chf (due to skewed distribution)
    df = log_transform(df, ["price_chf", "min_ord_val_chf"])
    # transform num ratings in to categorical variable
    df['num_ratings_cat'] = pd.qcut(df['num_ratings'], q=3, labels=['Low', 'Medium', 'High'])

    ########################################################################################################################
    # Research Question 1
    # Comparison of Pizza Prices between Cities (Kruskal-Wallis, Posthoc Dunn's Test)
    ########################################################################################################################
    # check variance
    variance = df.groupby('city')['log_price_chf'].var()
    print(f"variance: {variance.round(2)}")

    # run Kruskal-Wallis test
    kruskal_result = kruskal_wallis(df, "log_price_chf", "city")
    print(f"Kruskal-Wallis Test: {kruskal_result}, p-value: {kruskal_result.pvalue}")

    # run post-hoc Dunn-Bonferroni test to find which cities differ from each other
    dunn_results = dunn_posthoc(df, "log_price_chf", "city")
    alpha = 0.05
    sig_results = dunn_results[dunn_results < alpha].stack().reset_index()
    print(f"Post Hoc Dunn-Bonferroni (p < 0.05):\n {sig_results}")

    # average log prices per city to find out in what direction the differences are
    city_avg_log_price = df.groupby('city')['log_price_chf'].mean().sort_values(ascending=False)
    print(city_avg_log_price.round(2))

    ########################################################################################################################
    # Visualization: Violin Plot
    ########################################################################################################################
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

    ########################################################################################################################
    # Research Question 2
    # What Factors influence a Restaurants Rating? (OLS Regression Model)
    ########################################################################################################################
    # fit OLS model
    num_vars = ['log_price_chf', 'log_min_ord_val_chf', 'Median Monthly Wage (CHF)']
    cat_vars = ['num_ratings_cat', 'cuisine', 'delivery_fee_chf_cat', 'city']
    ols_model = fit_ols_model(df, 'rating', num_vars, cat_vars)
    print(ols_model.summary())

    ########################################################################################################################
    # Visualization: Facet Grid of Scatterplots num_rating vs. rating
    ########################################################################################################################
    # Facet grid for scatterplot num_rating vs. rating -> lmplot
    plt.figure(figsize=(14, 8))
    sns.lmplot(x='num_ratings', y='rating', data=df, hue='num_ratings_cat',
               markers=['o', 's', 'D'], palette='viridis', aspect=1.5, height=6)
    plt.title('Linear Regression of Ratings vs. Number of Ratings', fontsize=16, weight='bold')
    plt.xlabel('Number of Ratings', fontsize=14)
    plt.ylabel('Restaurant Rating', fontsize=14)
    plt.tight_layout()
    plt.show()

    """
    Visualizations that didn't make it into the documentation
    # Facet grid for scatterplot num_rating vs. rating -> FacetGrid
    g = sns.FacetGrid(df, col="num_ratings_cat", height=5, aspect=1.2)
    g.map(sns.regplot, 'num_ratings', 'rating', scatter_kws={'s': 10}, line_kws={"color": "orange"})
    g.set_axis_labels('Number of Ratings', 'Restaurant Rating')
    plt.tight_layout()
    plt.show()
    
    # city vs. rating (boxplot)
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='city', y='rating', data=df, palette='Set2')
    plt.title('Effect of City on Rating', fontsize=16, weight='bold')
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Restaurant Rating', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # median wage vs. rating (scatterplot)
    sns.scatterplot(x='Median Monthly Wage (CHF)', y='rating', data=df, hue='num_ratings_cat', palette='viridis')
    sns.regplot(x='Median Monthly Wage (CHF)', y='rating', data=df, scatter=False, color='red')
    plt.title('Effect of Median Wage on Rating', fontsize=16, weight='bold')
    plt.xlabel('Median Monthly Wage (CHF)', fontsize=14)
    plt.ylabel('Restaurant Rating', fontsize=14)
    plt.tight_layout()
    plt.show()
    """
if __name__ == "__main__":
    main()