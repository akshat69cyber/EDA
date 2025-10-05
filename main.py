import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("housing_data.csv")
print(df.columns)



print(df.isnull().sum())
df = df.dropna()
df = df.drop_duplicates()
print(df.dtypes)



print(df.head())
print(df.info())


# Example: Price distribution
# ----------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice'], kde=True, bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("SalePrice")
plt.ylabel("Number of Houses")
plt.show()
plt.figure(figsize=(8,5))
sns.boxplot(x=df['SalePrice'])
plt.title("Boxplot of House Prices")
plt.show()
sns.histplot(df['GrLivArea'], kde=True)
sns.boxplot(x=df['BedroomAbvGr'])
sns.boxplot(x=df['BedroomAbvGr'])


# ----------------------------------------------------------------------------------------------------------------------


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title("Living Area vs Sale Price")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(x='BedroomAbvGr', y='SalePrice', data=df)
plt.title("Bedrooms vs Sale Price")
plt.show()


if 'FullBath' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='FullBath', y='SalePrice', data=df)
    plt.title("Bathrooms vs Sale Price")
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------

df['Price_per_sqft'] = df['SalePrice'] / df['GrLivArea']


if 'YearBuilt' in df.columns:
    df['House_Age'] = 2025 - df['YearBuilt']


print(df[['SalePrice', 'GrLivArea', 'Price_per_sqft']].head())


plt.figure(figsize=(8,5))
sns.histplot(df['Price_per_sqft'], kde=True)
plt.title("Distribution of Price per Square Foot")
plt.show()



# ----------------------------------------------------------------------------------------------------------------------

# Average Sale Price per Year Sold
yearly_price = df.groupby('YrSold')['SalePrice'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.lineplot(x='YrSold', y='SalePrice', data=yearly_price, marker='o')
plt.title("Average House Price by Year Sold")
plt.xlabel("Year Sold")
plt.ylabel("Average Sale Price")
plt.grid(True)
plt.show()

# Average Sale Price per Month Sold
monthly_price = df.groupby('MoSold')['SalePrice'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(x='MoSold', y='SalePrice', data=monthly_price, palette='cool')
plt.title("Average House Price by Month Sold")
plt.xlabel("Month Sold")
plt.ylabel("Average Sale Price")
plt.show()

# Relation between Year Built and Sale Price
plt.figure(figsize=(8,5))
sns.scatterplot(x='YearBuilt', y='SalePrice', data=df)
plt.title("Sale Price vs Year Built")
plt.xlabel("Year Built")
plt.ylabel("Sale Price")
plt.show()


# ----------------------------------------------------------------------------------------------------------------------

# Check how certain features affect SalePrice
amenities = ['GarageCars', 'Fireplaces', 'PoolArea', 'WoodDeckSF', 'OpenPorchSF']

for feature in amenities:
    if feature in df.columns:
        plt.figure(figsize=(8,5))
        sns.scatterplot(x=df[feature], y=df['SalePrice'])
        plt.title(f"Sale Price vs {feature}")
        plt.xlabel(feature)
        plt.ylabel("Sale Price")
        plt.show()

# Neighborhood-wise average price
plt.figure(figsize=(12,6))
avg_neigh = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
sns.barplot(x=avg_neigh.index, y=avg_neigh.values)
plt.title("Average Sale Price by Neighborhood")
plt.xticks(rotation=90)
plt.xlabel("Neighborhood")
plt.ylabel("Average Sale Price")
plt.show()

# Overall Quality impact
plt.figure(figsize=(8,5))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title("Impact of Overall Quality on Sale Price")

# Correlation Heatmap - Final Insights
plt.figure(figsize=(12,8))

# Keep only numeric columns safely
numeric_df = df.select_dtypes(include=[np.number])

# Convert all numeric columns to numbers (ignore errors)
numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')

# Drop columns with all NaN values (after conversion)
numeric_df = numeric_df.dropna(axis=1, how='all')

# Compute correlation
corr = numeric_df.corr()

# Plot heatmap
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# # Exploratory Data Analysis for Real Estate Pricing
# This project performs EDA on a housing dataset to uncover factors influencing house prices.
# Steps include: Data Cleaning, Univariate & Multivariate Analysis, Feature Engineering, Market Trends, Customer Preferences, and Correlation Analysis.

