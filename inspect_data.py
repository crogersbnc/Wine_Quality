import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/WineQT.csv')
df.columns = df.columns.str.replace(' ', '_')

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Plot Histogram of each variable.
# Print Skew and Kurtosis for each Variable.

def inspect_variabes(df):

    for col in df.columns:
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Histogram for {col}")
        plt.show()
        print("--------------------------------")
        print(f"Skewness for {col}: {df[col].skew()}")

        if df[col].skew() > 0:
            s_pole = "positive "
        elif df[col].skew() < 0:
            s_pole = "negative "
        else:
            s_pole = "(somehow zero)"

        if -0.5 <= df[col].skew() <= 0.5:
            print(f"Symmetrical distribution")
        elif (-1 <= df[col].skew() < -0.5) or (0.5 < df[col].skew() <= 1):
            print(f"Moderate {s_pole}skew")
        else:
            print(f"High {s_pole}skew")

        print(f"Kurtosis for {col}: {df[col].kurtosis()}")
        if df[col].kurtosis() > 0:
            k_pole = "positive "
        elif df[col].kurtosis() < 0:
            k_pole = "negative "
        else:
            k_pole = "(somehow zero)"

        if -0.5 <= df[col].kurtosis() <= 0.5:
            print(f"Normal kurtosis")
        elif (-1 <= df[col].kurtosis() < -0.5) or (0.5 < df[col].kurtosis() <= 1):
            print(f"Moderate {s_pole}kurtosis")
        else:
            print(f"High {s_pole}kurtosis")
        print("--------------------------------\n")

inspect_variabes(df)





