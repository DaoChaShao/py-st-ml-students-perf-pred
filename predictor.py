#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/27 16:37
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from pandas import DataFrame, Series, read_csv
from pathlib import Path
from random import randint
from torch import load, no_grad, mean as torch_mean, abs as torch_abs

from src.nets.regression import RegressionTorchModel
from src.utils.config import CONFIG
from src.utils.helper import Timer
from src.utils.highlighter import red, green
from src.utils.PT import df2tensor
from src.utils.stats import (get_correlation_btw_Xy,
                             create_data_transformer, transform_data,
                             pca_importance,
                             split_data)


def preprocess_data():
    with Timer("Data Preprocessing"):
        # Set data path
        base: Path = Path(CONFIG.FILEPATHS.DATA_CSV)
        raw: DataFrame = read_csv(base, sep=";")
        # print(raw.head())
        # print(raw.shape)

        # Separate features and labels
        X: DataFrame = raw.iloc[:, : -1]
        y: Series = raw.iloc[:, -1]
        # print(X.shape, y.shape)
        assert X.shape[0] == y.shape[0], "Features and labels have different number of samples."

        # Get the correlations
        correlations: list[str] = get_correlation_btw_Xy(X, y, top_n=30, threshold=0.1)
        # print(correlations)
        # print(len(correlations))

        # Transform Data
        X = X[correlations]
        transformer = create_data_transformer(X)
        X = transform_data(X, transformer)
        # print(X.head())
        # print(X.shape)

        # Get the PCA importance
        importance, _, _ = pca_importance(X)
        print(importance)
        print(len(importance))

        # Select features based on correlations and PCA importance
        X = X[importance]
        # print(X)
        # print(X.shape)
        y: DataFrame = y.to_frame()
        # print(y)
        # print(y.shape)

        # Split Data
        _, _, X_prove, _, _, y_prove = split_data(
            X, y,
            randomness=CONFIG.PREPROCESSOR.RANDOM_STATE,
            shuffle_status=CONFIG.PREPROCESSOR.SHUFFLE_STATUS,
        )
        # print(X_prove.head())
        # print(y_prove.head())
        # print(X_prove.shape, y_prove.shape)

        return X_prove, y_prove


def main() -> None:
    """ Main Function """
    with Timer("Prove Data Prediction"):
        X, y = preprocess_data()

        # Convert DataFrame to Tensor
        X = df2tensor(X, accelerator=CONFIG.HYPERPARAMETERS.ACCELERATOR)
        y = df2tensor(y, accelerator=CONFIG.HYPERPARAMETERS.ACCELERATOR)
        # print(X.shape, y.shape)

        # Get the input shape of data
        index: int = randint(0, len(X) - 1)
        X = X[index].unsqueeze(0)
        y = y[index]
        # print(X)
        print(y)
        # print(X.shape, y.shape)
        feature_size: int = X.shape[1]
        # print(feature_size)

        # Load the trained model
        model = RegressionTorchModel(
            features=feature_size,
            hidden_units=CONFIG.MLP_PARAMS.HIDDEN_UNITS,
            out_size=1,
            dropout_rate=CONFIG.MLP_PARAMS.DROPOUT_RATE,

        )
        dict_state = load(CONFIG.FILEPATHS.SAVED_MODEL)
        model.load_state_dict(dict_state)
        model.eval()
        print("Model loaded successfully.")

        # Make predictions
        with no_grad():
            y_pred = model(X)
            print(y_pred)

        # Calculate error
        err = torch_abs(y_pred - y)
        out = green("Good") if err.item() <= 3 else red("Bad")
        print(f"Prediction Quality: {out}")


if __name__ == "__main__":
    main()
