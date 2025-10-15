import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load data
df = pd.read_csv('data/WineQT.csv')
df.columns = df.columns.str.replace(' ', '_')

# Features: All chemical properties
X = df.drop(['Id', 'quality'], axis=1)
# Target: Wine quality
y = df['quality']

# Split for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("--- data Split Summary ---")
print(f"X_train size: {X_train.shape}")
print(f"X_test size: {X_test.shape}")
print("-" * 30)

X_train.to_csv('X_train.csv', index=False)
# Define highly skewed features
skewed_features = [
    'residual_sugar',
    'chlorides',
    'sulphates',
    'total_sulfur_dioxide',
    'free_sulfur_dioxide',
    'fixed_acidity'
]

print(f"Applying Logarithmic Transformation to: {skewed_features}...")

# log(1+x) transform skewed features
for feature in skewed_features:

    X_train[feature] = np.log1p(X_train[feature])
    # Test transformed separately to prevent leakage.
    X_test[feature] = np.log1p(X_test[feature])



scaler = StandardScaler()

# Fit to training data only
scaler.fit(X_train)

# Transform both separately
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Recreate DataFrames for model use
feature_names = X.columns
X_train_processed = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
X_test_processed = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)


print("\n--- Final processed_data (First 5 Rows of Training Set) ---")
print(X_train_processed.head())
print("-" * 30)


# Save the processed feature data
X_train_processed.to_csv('X_train_processed.csv', index=False)
X_test_processed.to_csv('X_test_processed.csv', index=False)

# Save the target data
y_train.to_csv('y_train.csv', index=False, header=True)
y_test.to_csv('y_test.csv', index=False, header=True)

print("Processed data saved to disk.")