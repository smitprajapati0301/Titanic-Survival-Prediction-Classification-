import pandas as pd

df = pd.read_csv("data/train.csv")

# Drop columns
df.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ],
    inplace=True
)

# Fill Age with median
df["Age"].fillna(
    df["Age"].median(),
    inplace=True
)

# Fill Embarked with mode
df["Embarked"].fillna(
    df["Embarked"].mode()[0],
    inplace=True
)

print("Missing Values After Cleaning:")
print(df.isnull().sum())

df.to_csv(
    "data/titanic_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved.")