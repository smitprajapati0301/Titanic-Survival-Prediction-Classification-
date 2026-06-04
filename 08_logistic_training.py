import pandas as pd
import pickle
import time

from sklearn.linear_model import LogisticRegression

X_train = pd.read_csv(
    "data/X_train_scaled.csv"
)

y_train = pd.read_csv(
    "data/y_train.csv"
)

model = LogisticRegression(
    max_iter=1000
)

start_time = time.time()
model.fit(
    X_train,
    y_train.values.ravel()
)

end_time = time.time()

with open(
    "data/logistic_model.pkl",
    "wb"
) as f:
    pickle.dump(
        model,
        f
    )

print("Logistic Regression Model Saved")

print(f'time: {end_time-start_time:6f}')