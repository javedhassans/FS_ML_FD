# src/train.py
import warnings

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
from dataloader import load_data


def train_model(params):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        predds = clf.predict(X_test)
        acc = accuracy_score(y_test, predds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")
        print(f"accuracy: {acc}")
        return acc


def main():
    # Example default params
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "criterion": "gini",
        "random_state": 42,
    }

    train_model(params)


if __name__ == "__main__":
    main()
