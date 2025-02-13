import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load training dataset
train_df = pd.read_csv("train.csv")

# Display first few rows
print(train_df.head())

# Check dataset info
print(train_df.info())

# Check for missing values
print("Missing Values in Train Data:\n", train_df.isna().sum())

# Handling missing values by filling with mean for numerical and mode for categorical
numerical_cols = train_df.select_dtypes(include=[np.number]).columns
categorical_cols = train_df.select_dtypes(include=['object']).columns

train_df[numerical_cols] = train_df[numerical_cols].fillna(train_df[numerical_cols].mean())
train_df[categorical_cols] = train_df[categorical_cols].fillna(train_df[categorical_cols].mode().iloc[0])

# Drop unnecessary columns (ID)
train_df = train_df.drop(columns=['Id'], errors='ignore')

# Define features (X) and target variable (y)
X_train = train_df.drop(columns=["SalePrice"])
y_train = train_df["SalePrice"]

# Convert categorical variables to numerical
X_train = pd.get_dummies(X_train, drop_first=True)

# Load test dataset
test_df = pd.read_csv("test.csv")

# Display first few rows of test data
print(test_df.head())

# Identify numerical columns excluding the target variable for the test dataset
numerical_cols_test = test_df.select_dtypes(include=[np.number]).columns

# Handling missing values in test data
test_df[numerical_cols_test] = test_df[numerical_cols_test].fillna(test_df[numerical_cols_test].mean())
test_df[categorical_cols] = test_df[categorical_cols].fillna(test_df[categorical_cols].mode().iloc[0])

# Drop ID column (but keep it for final submission)
test_ids = test_df["Id"]
test_df = test_df.drop(columns=["Id"], errors='ignore')

# Convert categorical variables to numerical in test set
test_df = pd.get_dummies(test_df, drop_first=True)

# Ensure train and test have the same feature columns
X_train, test_df = X_train.align(test_df, join='left', axis=1, fill_value=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
test_df = scaler.transform(test_df)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)  # Predictions on training data
y_pred_test = model.predict(test_df)   # Predictions on test data

# Evaluate model performance on training data
mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred_train)

# Display metrics
print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R2): {r2:.2f}")

# Save predictions to CSV
submission_df = pd.DataFrame({'Id': test_ids, 'SalePrice': y_pred_test})
submission_df.to_csv('submission.csv', index=False)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted SalePrice')
plt.show()
