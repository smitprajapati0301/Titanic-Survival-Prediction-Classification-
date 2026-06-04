import pandas as pd

df = pd.read_csv(
    "data/titanic_encoded.csv"
)

# Features
X = df.drop(
    "Survived",
    axis=1
)

# Save features only
X.to_csv(
    "data/titanic_features.csv",
    index=False
)

# Save target separately
y = df["Survived"]

y.to_csv(
    "data/titanic_target.csv",
    index=False
)

print("Features saved.")
print("Target saved.")