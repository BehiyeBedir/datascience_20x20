import pandas as pd

# Load dataset from Kaggle input path
file_path = '/kaggle/input/life-expectancy-dataset/Life Expectancy Data.csv'
df = pd.read_csv(file_path)

# Fill missing numeric values with column mean
numeric_cols = df.select_dtypes(include='number').columns
for col in numeric_cols:
    mean_val = df[col].mean()
    df[col] = df[col].fillna(mean_val)

# Fill missing categorical values with column mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    mode_val = df[col].mode()
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val[0])

# Check total number of missing values after filling
print("Total missing values:", df.isnull().sum().sum())

# Print all column names with their lengths for inspection
for col in df.columns:
    print(f"'{col}' (length: {len(col)})")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Define target variable and features
target = 'Life expectancy '
X = df.drop(columns=[target])  
y = df[target]                 

# Select only numeric features for modeling
X = X.select_dtypes(include='number')

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Initialize Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on training data
rf_model.fit(X_train, y_train)

# Predict target variable on test data
y_pred = rf_model.predict(X_test)

# Calculate Root Mean Squared Error (RMSE) and R² score as evaluation metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² score: {r2:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

# Plot Actual vs Predicted life expectancy scatter plot with reference line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')   
plt.xlabel('Actual Life Expectancy')
plt.ylabel('Predicted Life Expectancy')
plt.title('Actual vs Predicted Life Expectancy')
plt.show()

# Plot residuals (errors) distribution histogram with KDE
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals (Actual - Predicted)')
plt.title('Residuals Distribution')
plt.show()

# Scatter plot showing relationship between Income composition and Life Expectancy, colored by Country Status
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Income composition of resources',
    y='Life expectancy ',
    hue='Status',
    alpha=0.7
)
plt.title('Relationship Between Income Composition and Life Expectancy', fontsize=14)
plt.xlabel('Income Composition of Resources')
plt.ylabel('Life Expectancy')
plt.legend(title='Country Status')
plt.grid(True)
plt.tight_layout()
plt.show()

# Line plot showing average Life Expectancy over the years
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x='Year',
    y='Life expectancy ',
    ci=None
)
plt.title('Average Life Expectancy Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Life Expectancy')
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plot showing relationship between Total Health Expenditure and Life Expectancy, colored by Country Status
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Total expenditure',
    y='Life expectancy ',
    hue='Status',
    alpha=0.7
)
plt.title('Relationship Between Total Health Expenditure and Life Expectancy')
plt.xlabel('Total Health Expenditure')
plt.ylabel('Life Expectancy')
plt.legend(title='Country Status')
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plot showing relationship between Alcohol Consumption and Life Expectancy, colored by Country Status
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Alcohol',
    y='Life expectancy ',
    hue='Status',
    alpha=0.7
)
plt.title('Relationship Between Alcohol Consumption and Life Expectancy')
plt.xlabel('Alcohol Consumption')
plt.ylabel('Life Expectancy')
plt.legend(title='Country Status')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate average under-five deaths per country and select top 10 countries with highest values
top_underfive = df.groupby('Country')['under-five deaths '].mean().sort_values(ascending=False).head(10)

# Bar plot of top 10 countries by average under-five deaths
plt.figure(figsize=(10, 6))
sns.barplot(x=top_underfive.values, y=top_underfive.index, palette='rocket')
plt.title('Top 10 Countries by Under-Five Deaths')
plt.xlabel('Average Under-Five Deaths')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
