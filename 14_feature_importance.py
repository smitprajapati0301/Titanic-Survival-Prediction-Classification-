import pandas as pd
import pickle

X_train = pd.read_csv(
    "data/X_train.csv"
)

with open(
    "data/rf_classifier.pkl",
    "rb"
) as f:
    model = pickle.load(f)

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)