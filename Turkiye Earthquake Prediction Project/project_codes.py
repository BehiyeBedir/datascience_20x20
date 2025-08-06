import pandas as pd

# Read the CSV file with ';' as the separator
df = pd.read_csv('/kaggle/input/turkey-earthquakes1915-2021/turkey_earthquakes(1915-2021).csv', sep=";")

# Print column names
print(df.columns)

# Display the first 5 rows
print(df.head())

# Show information about the DataFrame (data types, missing values, etc.)
print(df.info())

# Get statistical summary of numerical columns
print(df.describe())

# Print column names as a list
print(df.columns.tolist())

import seaborn as sns
import matplotlib.pyplot as plt

# Convert the 'Olus tarihi' column to datetime format
df['Olus tarihi'] = pd.to_datetime(df['Olus tarihi'], errors='coerce', dayfirst=True)

# Extract the year from the date
df['Yıl'] = df['Olus tarihi'].dt.year

# Group by year and count the number of earthquakes
yearly_counts = df.groupby('Yıl').size().reset_index(name='Deprem Sayısı')

# Plot the number of earthquakes per year
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_counts, x='Yıl', y='Deprem Sayısı', marker='o')
plt.title('Number of Earthquakes in Turkey by Year')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.grid(True)
plt.show()

# Calculate the average magnitude per year
avg_magnitude = df.groupby('Yıl')['Mw'].mean().reset_index()

# Plot average magnitude over the years
plt.figure(figsize=(12,6))
sns.lineplot(data=avg_magnitude, x='Yıl', y='Mw', marker='o')
plt.title('Average Earthquake Magnitude in Turkey by Year')
plt.xlabel('Year')
plt.ylabel('Average Magnitude (Mw)')
plt.grid(True)
plt.show()

# Calculate average depth per year
avg_depth = df.groupby('Yıl')['Derinlik'].mean().reset_index()

# Plot average depth over the years
plt.figure(figsize=(12,6))
sns.lineplot(data=avg_depth, x='Yıl', y='Derinlik', marker='o')
plt.title('Average Earthquake Depth in Turkey by Year')
plt.xlabel('Year')
plt.ylabel('Average Depth (km)')
plt.grid(True)
plt.show()

# Scatter plot: Magnitude vs Depth
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Derinlik', y='Mw', alpha=0.5)
plt.title('Earthquake Magnitude vs Depth in Turkey')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude (Mw)')
plt.grid(True)
plt.show()

# Remove rows with missing values in 'Mw' or 'Derinlik'
df_clean = df[['Mw', 'Derinlik']].dropna()

# Calculate correlation between magnitude and depth
correlation = df_clean['Mw'].corr(df_clean['Derinlik'])
print(f"Correlation coefficient between earthquake magnitude (Mw) and depth: {correlation:.3f}")

# Extract month from date
df['Ay'] = df['Olus tarihi'].dt.month

# Group by year and month to count earthquakes
monthly_counts = df.groupby(['Yıl', 'Ay']).size().reset_index(name='Deprem Sayısı')

# Pivot table for heatmap: rows = Year, columns = Month
pivot_table = monthly_counts.pivot_table('Deprem Sayısı', 'Yıl', 'Ay')

# Heatmap of earthquake counts by month and year
plt.figure(figsize=(12,8))
sns.heatmap(pivot_table, cmap='YlOrRd', linewidths=0.5)
plt.title('Monthly Earthquake Counts in Turkey (Year vs Month)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.show()

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Convert date column again just to be safe
df['Olus tarihi'] = pd.to_datetime(df['Olus tarihi'], dayfirst=True)

# Extract year again
df['Yıl'] = df['Olus tarihi'].dt.year

# Count number of earthquakes per year (Series format)
yearly_counts = df.groupby('Yıl').size()
print(type(yearly_counts))  # Check data type

# ADF test for stationarity
result = adfuller(yearly_counts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Apply first-order differencing to make data stationary
yearly_counts_diff = yearly_counts.diff().dropna()

# ADF test again on differenced data
result_diff = adfuller(yearly_counts_diff)
print('ADF Statistic after differencing:', result_diff[0])
print('p-value after differencing:', result_diff[1])

# Plot ACF and PACF to determine ARIMA order
plt.figure(figsize=(12,5))
plt.subplot(121)
plot_acf(yearly_counts_diff, ax=plt.gca(), lags=20)
plt.title('ACF Plot')
plt.subplot(122)
plot_pacf(yearly_counts_diff, ax=plt.gca(), lags=20)
plt.title('PACF Plot')
plt.show()

# Ensure index is of type int for ARIMA
yearly_counts.index = yearly_counts.index.astype(int)

# Fit ARIMA model (order determined from ACF/PACF analysis)
model = ARIMA(yearly_counts, order=(2,1,2))
model_fit = model.fit()

# Predict values within the available data
start = 0
end = len(yearly_counts) - 1
pred = model_fit.predict(start=start, end=end, typ='levels')
pred.index = yearly_counts.index

# Forecast future values (next 10 years)
forecast = model_fit.get_forecast(steps=10)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Create index for future years
last_year = int(yearly_counts.index[-1])
future_years = list(range(last_year + 1, last_year + 11))

# Plot actual values, predictions, and forecasts
plt.figure(figsize=(12,6))
plt.plot(yearly_counts.index, yearly_counts, label='Actual')
plt.plot(pred.index, pred, label='In-sample prediction')
plt.plot(future_years, forecast_mean, label='Forecast', linestyle='--')
plt.fill_between(future_years, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()
