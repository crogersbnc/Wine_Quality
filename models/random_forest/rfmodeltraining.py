from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# Load data
X_train_processed = pd.read_csv("processed_data/X_train_processed.csv")
X_test_processed = pd.read_csv("processed_data/X_test_processed.csv")
y_train = pd.read_csv("processed_data/y_train.csv").squeeze()
y_test = pd.read_csv("processed_data/y_test.csv").squeeze()

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model using the processed training data
print("Training the Random Forest Regressor...")
rf_model.fit(X_train_processed, y_train)
print("Training complete.")

# Predict
y_pred = rf_model.predict(X_test_processed)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print(f"\n--- Model Evaluation ---")
print(f"Random Forest RMSE on Test Set: {rmse:.4f}")

# Create a series of feature importance and sort them
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train_processed.columns
).sort_values(ascending=False)

print("\n--- Feature Importance Ranking ---")
print(feature_importance)

#Generate a horizontal bar plot of feature importance.
plt.figure(figsize=(10, 6))
plt.title("Feature Importance for Wine Quality Prediction")
plt.barh(feature_importance.index, feature_importance.values, color='indianred')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.tight_layout()
plt.show()
