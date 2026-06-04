import pandas as pd

df = pd.read_csv("data/train.csv")

print("Shape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())