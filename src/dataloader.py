# src/dataloader.py
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()

    # Create a DataFrame from the data
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add the target column to the DataFrame
    df["target"] = data.target

    # Convert the DataFrame to a NumPy array
    X = df.drop("target", axis=1).values
    y = df["target"].values

    return X, y
