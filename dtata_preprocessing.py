import pandas as pd


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    X = df[["hours", "sleep", "attendance"]]
    y = df["pass"]
    return X, y
