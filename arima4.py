import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import itertools
from math import sqrt

# Load the dataset
df = pd.read_csv('Walmart.csv')

# Display the first few rows of the dataframe to ensure it loaded correctly
df.info()

# Function to parse dates with different formats
def parse_dates(date):
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(date, format=fmt)
        except ValueError:
            pass
    return pd.NaT

# Apply the function to the Date column
df['Date'] = df['Date'].apply(parse_dates)

# Check for missing values again
missing_values = df.isnull().sum()
with open('missing_values.txt', 'w') as f:
    f.write(str(missing_values))

df.describe().to_csv('data_summary.csv')

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Extract the correlation with Weekly_Sales
correlation_with_sales = correlation_matrix["Weekly_Sales"].sort_values(ascending=False)
with open('correlation_with_sales.txt', 'w') as f:
    f.write(str(correlation_with_sales))

# Set the style for the plots
sns.set(style="whitegrid")

# List of variables to plot against Weekly_Sales
variables = ['Holiday_Flag', 'Fuel_Price', 'Temperature', 'CPI', 'Unemployment', 'Store']

# Create a scatter plot for each variable against Weekly_Sales
plt.figure(figsize=(15, 10))
for i, var in enumerate(variables, start=1):
    plt.subplot(2, 3, i)
    sns.scatterplot(data=df, x=var, y='Weekly_Sales', alpha=0.5)
    plt.title(f'Weekly Sales vs {var}')
    plt.xlabel(var)
    plt.ylabel('Weekly Sales')
plt.tight_layout()
plt.savefig('scatter_plots.png')
plt.close()

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