import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

scaled = pd.read_csv('X_train_processed.csv')
unscaled = pd.read_csv('processed_data/X_train.csv')

skewed_features = [
    'residual_sugar',
    'chlorides',
    'sulphates',
    'total_sulfur_dioxide',
    'free_sulfur_dioxide',
    'fixed_acidity'
]

def compare_plot(scaled, unscaled, features):
    for feature in features:
        transformed_data = scaled[feature]
        original_data = unscaled[feature]

        plt.figure(figsize=(14,10))
        plt.suptitle(f"Distribution Check: '{feature}' Feature Transformation", fontsize=12)
        plt.subplot(1, 2, 1)
        sns.histplot(original_data, bins=30, kde=True, color='red')
        plt.title("Before Transformation (Raw data)")
        plt.xlabel(f"Original {feature} Value")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sns.histplot(transformed_data, bins=30, kde=True, color='darkblue')
        plt.title("After Log Transformation and Standard Scaling")
        plt.xlabel("Processed Value (Mean=0, Std=1)")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

compare_plot(scaled, unscaled, skewed_features)

