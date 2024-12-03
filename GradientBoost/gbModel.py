import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb

train_data = pd.read_csv('TRAIN_PCA_STANDARDIZED.csv')
test_data = pd.read_csv('TEST_PCA_STANDARDIZED.csv')

X_train = train_data.iloc[:, 2:] 
y_train_away = train_data.iloc[:, 0] 
y_train_home = train_data.iloc[:, 1]

X_test = test_data.iloc[:, 2:]
y_test_away = test_data.iloc[:, 0]
y_test_home = test_data.iloc[:, 1]

home_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

away_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=10,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

home_model.fit(X_train, y_train_home)
away_model.fit(X_train, y_train_away)

y_pred_home = home_model.predict(X_test)
y_pred_away = away_model.predict(X_test)
y_pred_winner = (y_pred_home > y_pred_away).astype(int)
y_test_winner = (y_test_home > y_test_away).astype(int)
win_accuracy = accuracy_score(y_test_winner, y_pred_winner)
print(f"Model Win Accuracy: {win_accuracy * 100:.2f}%")

home_mse = mean_squared_error(y_test_home, y_pred_home)
away_mse = mean_squared_error(y_test_away, y_pred_away)

print(f"Home Score Prediction Mean Squared Error: {home_mse:.2f}")
print(f"Away Score Prediction Mean Squared Error: {away_mse:.2f}")

# Plot 1: Home Scores
plt.figure(figsize=(10, 6))
plt.scatter(y_test_home, y_pred_home, alpha=0.7, color='green')
plt.plot([y_test_home.min(), y_test_home.max()], [y_test_home.min(), y_test_home.max()], color='red', linestyle='--', linewidth=2)
plt.title('Home Scores: Predicted vs Actual')
plt.xlabel('Actual Home Score')
plt.ylabel('Predicted Home Score')
plt.grid()
plt.show()

# Plot 2: Away Scores
plt.figure(figsize=(10, 6))
plt.scatter(y_test_away, y_pred_away, alpha=0.7, color='green')
plt.plot([y_test_away.min(), y_test_away.max()], [y_test_away.min(), y_test_away.max()], color='red', linestyle='--', linewidth=2)
plt.title('Away Scores: Predicted vs Actual')
plt.xlabel('Actual Away Score')
plt.ylabel('Predicted Away Score')
plt.grid()
plt.show()

# Plot 3: Total Scores
y_test_total = y_test_home + y_test_away
y_pred_total = y_pred_home + y_pred_away

plt.figure(figsize=(10, 6))
plt.scatter(y_test_total, y_pred_total, alpha=0.7, color='green')
plt.plot([y_test_total.min(), y_test_total.max()], [y_test_total.min(), y_test_total.max()], color='red', linestyle='--', linewidth=2)
plt.title('Total Scores: Predicted vs Actual')
plt.xlabel('Actual Total Score')
plt.ylabel('Predicted Total Score')
plt.grid()
plt.show()

home_residuals = y_test_home - y_pred_home
away_residuals = y_test_away - y_pred_away

# Residual Plot for Home Scores
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_home, home_residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals for Home Score Predictions')
plt.xlabel('Predicted Home Score')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid()
plt.show()

# Residual Plot for Away Scores
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_away, away_residuals, alpha=0.7, color='green')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals for Away Score Predictions')
plt.xlabel('Predicted Away Score')
plt.ylabel('Residual (Actual - Predicted)')
plt.grid()
plt.show()