# src/tune_hyperparameter
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from train import train_model

# Define search space
search_space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 150, 200]),
    "max_depth": hp.choice("max_depth", [3, 5, 10, 15, 20]),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "random_state": 42,
}


# Define objective function
def objective(params):
    with mlflow.start_run(nested=True):
        acc = train_model(params)
        loss = 1 - acc  # minimize loss
        return {"loss": loss, "status": STATUS_OK}


def run_tuning():
    with mlflow.start_run(run_name="tuning_experiment"):
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )
        print("Best params:", best)
