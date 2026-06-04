import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier

X_train = pd.read_csv(
    "data/X_train.csv"
)

y_train = pd.read_csv(
    "data/y_train.csv"
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train.values.ravel()
)

with open(
    "data/rf_classifier.pkl",
    "wb"
) as f:
    pickle.dump(
        model,
        f
    )

print("Random Forest Model Saved")