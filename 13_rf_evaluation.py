import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

y_test = pd.read_csv(
    "data/y_test.csv"
)

predictions = pd.read_csv(
    "data/rf_predictions.csv"
)

print("\nRandom Forest Results")
print("-" * 35)

print(
    "Accuracy  :",
    round(
        accuracy_score(
            y_test,
            predictions
        ),
        4
    )
)

print(
    "Precision :",
    round(
        precision_score(
            y_test,
            predictions
        ),
        4
    )
)

print(
    "Recall    :",
    round(
        recall_score(
            y_test,
            predictions
        ),
        4
    )
)

print(
    "F1 Score  :",
    round(
        f1_score(
            y_test,
            predictions
        ),
        4
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)