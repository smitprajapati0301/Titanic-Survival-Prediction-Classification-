import pandas as pd

df = pd.read_csv(
    "data/titanic_cleaned.csv"
)

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=[
        "Sex",
        "Embarked"
    ],
    dtype=int
)

df.to_csv(
    "data/titanic_encoded.csv",
    index=False
)

print("Encoded dataset saved.")