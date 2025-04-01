# Load all libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
import itertools

# Load train dataset
df = pd.read_csv('train.csv')
# Load features dataset and join it with train data
features_df = pd.read_csv('features.csv')
df = pd.merge(df, features_df.drop(['IsHoliday'], axis = 1), how = 'left', on = ['Store', 'Date'])
# Load store dataset and join with above data
stores_df = pd.read_csv('stores.csv')
df = pd.merge(df, stores_df, how = 'left', on = ['Store'])

# Filter data for department 92 of store 20
df = df[(df['Store'] == 20) & (df['Dept'] == 92)]

df.shape
df.head().to_csv('filtered_data_head.csv')

# Let's explore variables, their data types, and total non-null values
df_info = df.info()
with open('df_info.txt', 'w') as f:
    f.write(str(df_info))

# Summary statistics of the dataset
df_summary = df[['Weekly_Sales', 'Temperature', 'CPI', 'Size']].describe()
df_summary.to_csv('data_summary.csv')

print('Min Date in Data is - {}'.format(df['Date'].min()))
print('Max Date in Data is - {}'.format(df['Date'].max()))

temp = pd.DataFrame(df.groupby('Type')['Store'].nunique()).reset_index()
print(temp)
plt.figure(figsize = (12,6))
plt.pie(temp['Store'], labels = temp['Type'], autopct = '%.0f%%')
plt.savefig('store_type_distribution.png')
plt.close()

# Size distribution of stores for each store type
plt.figure(figsize = (12,8))
sns.boxplot(x = 'Type', y ='Size', data = df, showfliers = False)
plt.savefig('size_distribution.png')
plt.close()

# Distribution of weekly sales based on store type
plt.figure(figsize = (12,8))
sns.boxplot(x = 'Type', y ='Weekly_Sales', data = df, showfliers = False)
plt.savefig('weekly_sales_distribution.png')
plt.close()

# Impact of holidays on weekly sales
plt.figure(figsize = (12,8))
sns.boxplot(x = 'IsHoliday', y ='Weekly_Sales', data = df, showfliers = False)
plt.savefig('holiday_impact.png')
plt.close()

feature_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
       'MarkDown5', 'CPI', 'Unemployment', 'Size']
plt.figure(figsize = (18,12))
sns.heatmap(df[feature_cols].corr(), annot = True)
plt.savefig('correlation_heatmap.png')
plt.close()

# Impute NULL values
df['MarkDown1'] = df['MarkDown1'].fillna(0)
df['MarkDown2'] = df['MarkDown2'].fillna(0)
df['MarkDown3'] = df['MarkDown3'].fillna(0)
df['MarkDown4'] = df['MarkDown4'].fillna(0)
df['MarkDown5'] = df['MarkDown5'].fillna(0)

# Create year, month, and date
df['Date'] = pd.to_datetime(df['Date'])
df['month_date'] = df['Date'].apply(lambda i : i.month)
df['day_date'] = df['Date'].apply(lambda i : i.day)
df['year_date'] = df['Date'].apply(lambda i : i.year)

# One hot encoding
cols_to_encode = ['Type', 'IsHoliday']
df = pd.get_dummies(data = df, columns = cols_to_encode, drop_first = True)

# Standard Scaler
standard_scaler = StandardScaler()
feature_cols = ['Temperature', 'Fuel_Price', 'MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'Size']
transformed_features = standard_scaler.fit_transform(df[feature_cols])

df[feature_cols] = transformed_features

# Time Series Analysis
sales_data = df.groupby('Date')['Weekly_Sales'].sum().sort_index()

# Perform the Augmented Dickey-Fuller test to check for stationarity
adf_test = adfuller(sales_data)
adf_p_value = adf_test[1]
with open('adf_test_results.txt', 'w') as f:
    f.write(f'ADF Statistic: {adf_test[0]}\n')
    f.write(f'p-value: {adf_p_value}\n')

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(sales_data, ax=plt.gca(), lags=40)
plt.title('Autocorrelation Function')
plt.subplot(122)
plot_pacf(sales_data, ax=plt.gca(), lags=40)
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.savefig('acf_pacf_plots.png')
plt.close()

# Define train and test sets
train_size = int(len(sales_data) * 0.8)
train = sales_data[:train_size]
test = sales_data[train_size:]

# Auto ARIMA to find best p, d, q
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

# Evaluate models
best_aic = float("inf")
best_pdq = None
best_model = None

for param in pdq:
    try:
        temp_model = ARIMA(train, order=param)
        temp_result = temp_model.fit()
        if temp_result.aic < best_aic:
            best_aic = temp_result.aic
            best_pdq = param
            best_model = temp_result
    except:
        continue

# Save summary of the best model
with open('best_model_summary.txt', 'w') as f:
    f.write("Best Model Summary:\n")
    f.write(str(best_model.summary()))
    f.write(f"\nBest pdq: {best_pdq}\n")

# Forecast using the best model
forecast_steps = len(test)
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_df = forecast.summary_frame()
forecast_values = forecast_df['mean']

# Calculate RMSE for the best model
best_rmse = sqrt(mean_squared_error(test, forecast_values))
with open('best_rmse.txt', 'w') as f:
    f.write(f"Best RMSE: {best_rmse}\n")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data', color='orange')
plt.plot(test.index, forecast_values, label='Forecast', color='green')
plt.fill_between(test.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='green', alpha=0.2)
plt.title('ARIMA Model Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.legend()
plt.savefig('forecast_plot.png')
plt.close()

# Model diagnostics: Plot residuals and perform tests
residuals = best_model.resid

# Plot residuals
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title('Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.subplot(212)
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('residuals_plot.png')
plt.close()