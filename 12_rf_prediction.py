import pandas as pd
import pickle

X_test = pd.read_csv(
    "data/X_test.csv"
)

with open(
    "data/rf_classifier.pkl",
    "rb"
) as f:
    model = pickle.load(f)

predictions = model.predict(
    X_test
)

pd.DataFrame(
    predictions,
    columns=["Predicted_Survival"]
).to_csv(
    "data/rf_predictions.csv",
    index=False
)

print("Predictions Saved")