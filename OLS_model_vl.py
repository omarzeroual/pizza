import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("data/pizza_final.csv", delimiter = ";", header = 0)

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

# create correlation heatmap
df_heatmap = df[['log_price_chf', 'log_min_ord_val_chf', 'Median Monthly Wage (CHF)', 'rating']]
corr_matrix = df_heatmap.corr()

plt.figure(figsize = (10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=" .2f",
            linewidths=0.5, cbar=True)
plt.title("Correlation Heatmap")
plt.show()

