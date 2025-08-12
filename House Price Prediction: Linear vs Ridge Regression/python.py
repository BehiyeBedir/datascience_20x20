import pandas as pd

# Load the training dataset from the Kaggle input path
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

# Display dataset information (number of columns, data types, null values, etc.)
print(df.info())

# List all column names
print(df.columns.tolist())

# Calculate missing values and their percentage for each column
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_data = pd.DataFrame({'MissingCount': missing, 'MissingPercent': missing_percent})

# Filter only columns with missing values and sort by percentage in descending order
missing_data = missing_data[missing_data['MissingCount'] > 0].sort_values(by='MissingPercent', ascending=False)
print(missing_data)

# Columns with too many missing values (will be dropped)
cols_to_drop = ['PoolQC', 'Alley', 'Fence', 'MiscFeature', 'FireplaceQu']

# Drop the selected columns from the dataframe
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")

# Fill missing values in numeric columns with the median
num_cols_with_na = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea']
for col in num_cols_with_na:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Fill missing values in categorical columns with the string 'None'
cat_cols_with_na = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                    'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond',
                    'BsmtQual', 'Electrical']
for col in cat_cols_with_na:
    df[col] = df[col].fillna('None')

# Convert categorical variables into dummy/indicator variables
# drop_first=True prevents the dummy variable trap
df = pd.get_dummies(df, drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

from sklearn.model_selection import train_test_split

# Split the dataset into training (80%) and validation (20%) sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)  # Train the model
y_pred_lr = lr.predict(X_valid)  # Predictions on the validation set
rmse_lr = np.sqrt(mean_squared_error(y_valid, y_pred_lr))  # Calculate RMSE

# --- 2. Ridge Regression (alpha=1) ---
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_valid)
rmse_ridge = np.sqrt(mean_squared_error(y_valid, y_pred_ridge))

# Print results
print(f"Linear Regression RMSE: {rmse_lr:.2f}")
print(f"Ridge Regression RMSE: {rmse_ridge:.2f}")

# --- 3. Visualization: Actual vs Predicted for Ridge Regression ---
plt.figure(figsize=(8,6))
plt.scatter(y_valid, y_pred_ridge, alpha=0.5)  # Scatter plot of actual vs predicted values
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')  # Reference line
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Ridge Regression: Actual vs Predicted')
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# List of alpha values to try
alphas = [0.01, 0.1, 1, 10, 100, 200, 500, 1000]
ridge = Ridge()
param_grid = {'alpha': alphas}

# Perform grid search with 5-fold cross-validation using negative MSE as the score
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best alpha value
print(f"Best alpha: {grid_search.best_params_['alpha']}")

# Retrain the model with the best alpha
best_ridge = grid_search.best_estimator_
y_pred_best = best_ridge.predict(X_valid)
rmse_best = np.sqrt(mean_squared_error(y_valid, y_pred_best))
print(f"Optimized Ridge RMSE: {rmse_best:.2f}")

import seaborn as sns

# Calculate residuals (errors between actual and predicted values)
residuals = y_valid - y_pred_best

plt.figure(figsize=(15,4))

# 1. Residual Plot
plt.subplot(1,3,1)
sns.scatterplot(x=y_pred_best, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# 2. Histogram of Residuals
plt.subplot(1,3,2)
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.title('Histogram of Residuals')

# 3. Prediction Error Plot
plt.subplot(1,3,3)
sns.scatterplot(x=y_valid, y=y_pred_best, alpha=0.5)
plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Prediction Error Plot')

plt.tight_layout()
plt.show()
