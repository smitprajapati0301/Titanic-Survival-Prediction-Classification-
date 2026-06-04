import pandas as pd
import pickle

X_test = pd.read_csv(
    "data/X_test_scaled.csv"
)

with open(
    "data/logistic_model.pkl",
    "rb"
) as f:
    model = pickle.load(f)

predictions = model.predict(
    X_test
)
print(X_test.head())
print(predictions[:5])
print(type(predictions))

pd.DataFrame(
    predictions,
    columns=["Predicted_Survival"]
).to_csv(
    "data/logistic_predictions.csv",
    index=False
)

print("Predictions Saved")