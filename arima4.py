# Load all libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
import itertools
import os

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Load train dataset
df = pd.read_csv('train.csv')
# Load features dataset and join it with train data
features_df = pd.read_csv('features.csv')
df = pd.merge(df, features_df.drop(['IsHoliday'], axis=1), how='left', on=['Store', 'Date'])
# Load store dataset and join with above data
stores_df = pd.read_csv('stores.csv')
df = pd.merge(df, stores_df, how='left', on=['Store'])

# Filter data for Store 20 (all departments)
df = df[df['Store'] == 20]

# Summary statistics of the dataset
df_summary = df[['Weekly_Sales', 'Temperature', 'CPI', 'Size']].describe()
df_summary.to_csv('outputs/data_summary.csv')

# Store type distribution
temp = pd.DataFrame(df.groupby('Type')['Store'].nunique()).reset_index()
print(temp)
plt.figure(figsize=(12, 6))
plt.pie(temp['Store'], labels=temp['Type'], autopct='%.0f%%')
plt.savefig('outputs/store_type_distribution.png')
plt.close()

# Size and sales distribution
plt.figure(figsize=(12, 8))
sns.boxplot(x='Type', y='Size', data=df, showfliers=False)
plt.savefig('outputs/size_distribution.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(x='Type', y='Weekly_Sales', data=df, showfliers=False)
plt.savefig('outputs/weekly_sales_distribution.png')
plt.close()

# Correlation heatmap
feature_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                'MarkDown5', 'CPI', 'Unemployment', 'Size']
plt.figure(figsize=(18, 12))
sns.heatmap(df[feature_cols].corr(), annot=True)
plt.savefig('outputs/correlation_heatmap.png')
plt.close()

# Impute NULL values
for col in ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']:
    df[col] = df[col].fillna(0)

# Create year, month, and date
df['Date'] = pd.to_datetime(df['Date'])
df['month_date'] = df['Date'].dt.month
df['day_date'] = df['Date'].dt.day
df['year_date'] = df['Date'].dt.year

# One-hot encoding
cols_to_encode = ['Type', 'IsHoliday']
df = pd.get_dummies(data=df, columns=cols_to_encode, drop_first=True)

# Standard Scaler
standard_scaler = StandardScaler()
transformed_features = standard_scaler.fit_transform(df[feature_cols])
df[feature_cols] = transformed_features

# Time Series Analysis and Forecasting by Department
departments = df['Dept'].unique()
all_forecasts = {}
train_data = {}
test_data = {}

# Define train/test split proportion
train_size_proportion = 0.8

for dept in departments:
    # Filter data for current department
    dept_data = df[df['Dept'] == dept]
    sales_data = dept_data.groupby('Date')['Weekly_Sales'].sum().sort_index()
    
    if len(sales_data) < 10:  # Skip if too few data points
        print(f"Skipping Dept {dept} - insufficient data ({len(sales_data)} points)")
        continue
    
    # Train-test split
    train_size = int(len(sales_data) * train_size_proportion)
    train = sales_data[:train_size]
    test = sales_data[train_size:]
    
    # Store for later use
    train_data[dept] = train
    test_data[dept] = test
    
    # Auto ARIMA to find best p, d, q
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    
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
    
    if best_model is None:
        print(f"Skipping Dept {dept} - no valid ARIMA model found")
        continue
    
    # Forecast
    forecast_steps = len(test)
    forecast = best_model.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    all_forecasts[dept] = forecast_df['mean']

# Aggregate forecasts across all departments
forecast_dates = test_data[list(test_data.keys())[0]].index  # Use test dates from first dept
aggregated_forecast = pd.Series(0, index=forecast_dates)

for dept, forecast_values in all_forecasts.items():
    aligned_forecast = forecast_values.reindex(forecast_dates, fill_value=0)
    aggregated_forecast += aligned_forecast

# Aggregate actual test data for comparison
aggregated_test = pd.Series(0, index=forecast_dates)
for dept, test in test_data.items():
    aligned_test = test.reindex(forecast_dates, fill_value=0)
    aggregated_test += aligned_test

# Aggregate train data for plotting
aggregated_train = pd.Series(0, index=train_data[list(train_data.keys())[0]].index)
for dept, train in train_data.items():
    aligned_train = train.reindex(aggregated_train.index, fill_value=0)
    aggregated_train += aligned_train

# Calculate performance metrics
mse = mean_squared_error(aggregated_test, aggregated_forecast)
rmse = sqrt(mse)
mae = mean_absolute_error(aggregated_test, aggregated_forecast)
mape = np.mean(np.abs((aggregated_test - aggregated_forecast) / aggregated_test)) * 100
r2 = r2_score(aggregated_test, aggregated_forecast)

# Save metrics to a file
with open('outputs/aggregated_performance_metrics.txt', 'w') as f:
    f.write(f"Performance Metrics for Aggregated Forecast (Store 20):\n")
    f.write(f"MSE: {mse:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"MAPE: {mape:.2f}%\n")
    f.write(f"R²: {r2:.4f}\n")

# Plot aggregated results
plt.figure(figsize=(12, 6))
plt.plot(aggregated_train.index, aggregated_train, label='Training Data')
plt.plot(aggregated_test.index, aggregated_test, label='Test Data', color='orange')
plt.plot(aggregated_forecast.index, aggregated_forecast, label='Aggregated Forecast', color='green')
plt.title('Aggregated ARIMA Forecast for Store 20 vs Actual')
plt.xlabel('Date')
plt.ylabel('Total Weekly Sales')
plt.legend()
plt.savefig('outputs/aggregated_forecast_plot.png')
plt.close()