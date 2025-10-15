import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



# Load data
X_train_processed = pd.read_csv("processed_data/X_train_processed.csv")
X_test_processed = pd.read_csv("processed_data/X_test_processed.csv")
print("data Loaded.")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

'''
Model Training
'''

# Linear Baseline
lr_model = LinearRegression()
print("Training Linear Regression...")
start_time_lr = time.time()
lr_model.fit(X_train_processed, y_train)
end_time_lr = time.time()
print(f"Linear Regression Training Time: {end_time_lr - start_time_lr:.4f} seconds")

# RF Primary
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
print("Training Random Forest Regressor...")
start_time_rf = time.time()
rf_model.fit(X_train_processed, y_train)
end_time_rf = time.time()
print(f"Random Forest Training Time: {end_time_rf - start_time_rf:.4f} seconds")

# Predict models for comparison later
y_pred_lr = lr_model.predict(X_test_processed)
y_pred_rf = rf_model.predict(X_test_processed)


'''
Model Evaluation
'''

# Helper function
def evaluate_model(y_true, y_pred, model_name):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} Evaluation ---")
    print(f"RMSE (Error in Quality Points): {rmse:.4f}")
    print(f"R-squared (Variance Explained): {r2:.4f}")
    return rmse

# Evaluate models
rmse_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression")
rmse_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")


'''
Feature Importance Ranking
'''

# Feature importance scores
importances = rf_model.feature_importances_

# Match scores to feature names and sort
feature_names = X_train_processed.columns
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n--- Feature Importance Ranking (Random Forest) ---")
print(feature_importance)

# Visualization
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh', color='indianred')
plt.title("Feature Importance for Wine Quality Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Chemical Feature")
plt.gca().invert_yaxis() # Highest importance at the top
plt.show()

'''
Hyperparameter Tuning
'''

# Define Parameters Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}

# Instantiate the Grid Search object
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1,
    n_jobs=-1 # Use all available CPU cores
)

# Search the training data
print("\nStarting Hyperparameter Tuning...")
grid_search.fit(X_train_processed, y_train)

# Get the best model and score
best_rf_model = grid_search.best_estimator_
best_rmse = np.sqrt(-grid_search.best_score_) # Convert to positive RMSE

print("\n--- Hyperparameter Tuning Results ---")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-Validated RMSE: {best_rmse:.4f}")

# Evaluation of the tuned model on the test set
y_pred_tuned = best_rf_model.predict(X_test_processed)
final_rmse = evaluate_model(y_test, y_pred_tuned, "Tuned Random Forest")

'''
Acccuracy Evaluation
'''

# Round the continuous predictions (y_pred_rf) to the nearest integer
# and clip them to the valid range [3, 8]
y_pred_int = np.round(y_pred_rf).astype(int)
y_pred_int = np.clip(y_pred_int, 3, 8) # Ensure scores are within the valid 3-8 range

# Calculate Exact Accuracy
exact_accuracy = accuracy_score(y_test, y_pred_int)
print(f"\nExact Match Accuracy: {exact_accuracy:.4f} ({exact_accuracy*100:.2f}%)")

# Calculate Accuracy within +/- 1 point
# Check if the absolute difference between predicted and actual is <= 1
accuracy_plus_minus_one = np.mean(np.abs(y_test - y_pred_int) <= 1)
print(f"Accuracy within +/- 1 Point: {accuracy_plus_minus_one:.4f} ({accuracy_plus_minus_one*100:.2f}%)")

np.savetxt('processed_data/y_pred_int.csv', y_pred_int, delimiter=',', fmt='%d')
np.savetxt('y_test.csv', y_test, delimiter=',', fmt='%d')



