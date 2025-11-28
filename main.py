#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/27 16:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :   

from pandas import DataFrame, Series, read_csv
from random import randint
from torch import nn, optim

from src.dataloader.general import TorchDataLoader
from src.datasets.label_regression import RegressionTorchDatasetForPrediction
from src.nets.regression import RegressionTorchModel
from src.trainers.regression import RegressionTorchTrainer
from src.utils.config import CONFIG
from src.utils.helper import Timer
from src.utils.PT import TorchRandomSeed
from src.utils.stats import (summary_dataframe,
                             split_data,
                             create_data_transformer, transform_data,
                             pca_importance, get_correlation_btw_Xy)


def preprocess_data():
    """ Preprocess Data """
    with Timer("Data Preprocessing"):
        # Set data path
        raw: DataFrame = read_csv(CONFIG.FILEPATHS.DATA_CSV, sep=";")
        # print(data)
        summary_dataframe(raw)

        # Separate features and labels
        X: DataFrame = raw.iloc[:, : -1]
        y: Series = raw.iloc[:, -1]
        # print(X.head())
        # print(y.head())
        # print(X.shape, y.shape)
        assert X.shape[0] == y.shape[0], "Features and labels have different number of samples."

        # Get correlation between features and labels
        correlations: list[str] = get_correlation_btw_Xy(X, y, top_n=30, threshold=0.1)
        # print(correlations)
        # print(len(correlations))

        # Transform Data
        X = X[correlations]
        transformer = create_data_transformer(X)
        X = transform_data(X, transformer)
        # print(X.head())
        # print(X.shape)

        # PCA Importance
        importance, _, _ = pca_importance(X)
        # print(importance)
        # print(len(importance))

        # Select features based on correlations and PCA importance
        X = X[importance]
        # print(X)
        # print(X.shape)
        y: DataFrame = DataFrame(y)
        # print(y)
        # print(y.shape)

        # Split data into training, validation, and test sets
        X_train, X_valid, _, y_train, y_valid, _ = split_data(
            X, y,
            randomness=CONFIG.PREPROCESSOR.RANDOM_STATE,
            shuffle_status=CONFIG.PREPROCESSOR.SHUFFLE_STATUS,
        )
        # print(len(X_train))
        # print(len(X_valid))

        return X_train, X_valid, y_train, y_valid


def prepare_data():
    with Timer("Data Preparation"):
        # preprocess_data()
        X_train, X_valid, y_train, y_valid = preprocess_data()

        # Create PyTorch Datasets
        train_dataset = RegressionTorchDatasetForPrediction(X_train, y_train)
        valid_dataset = RegressionTorchDatasetForPrediction(X_valid, y_valid)
        # its: int = randint(0, len(train_dataset) - 1)
        # print(f"Train Sample {its:>05}: Label={train_dataset.labels[its]}, Features={train_dataset.features[its]}")
        # print(f"Train Sample {its:>05}: X shape={X_train.iloc[its].shape}, y shape={y_train.iloc[its].shape}")
        # ivs = randint(0, len(valid_dataset) - 1)
        # print(f"Valid Sample {ivs:>05}: Label={valid_dataset.labels[ivs]}, Features={valid_dataset.features[ivs]}")
        # print(f"Valid Sample {ivs:>05}: X shape={X_valid.iloc[ivs].shape}, y shape={y_valid.iloc[ivs].shape}")

        # Create DataLoaders
        train_loader = TorchDataLoader(
            dataset=train_dataset,
            batch_size=CONFIG.PREPROCESSOR.BATCHES,
            is_shuffle=CONFIG.PREPROCESSOR.SHUFFLE_STATUS,
        )
        valid_loader = TorchDataLoader(
            dataset=valid_dataset,
            batch_size=CONFIG.PREPROCESSOR.BATCHES,
            is_shuffle=CONFIG.PREPROCESSOR.SHUFFLE_STATUS,
        )
        # itl: int = randint(0, len(train_loader) - 1)
        # print(f"Train Loader Batch {itl:>05}: {train_loader[itl]}")
        # ivl: int = randint(0, len(valid_loader) - 1)
        # print(f"Valid Loader Batch {ivl:>05}: {valid_loader[ivl]}")

        return train_loader, valid_loader


def main() -> None:
    """ Main Function """
    with TorchRandomSeed("Student Performance Prediction"):
        # prepare_data()
        train_loader, valid_loader = prepare_data()
        # print(train_loader[0])

        # Get the feature size for model input
        feature_size: int = train_loader[0][0].shape[0]
        # print(f"Feature Size: {feature_size}")

        # Initialize the Regression Model
        model = RegressionTorchModel(
            features=feature_size,
            hidden_units=CONFIG.MODEL_PARAMS.HIDDEN_UNITS,
            out_size=1,
            dropout_rate=CONFIG.MODEL_PARAMS.DROPOUT_RATE,
        )
        model.summary()

        # Set optimizer and loss function
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG.HYPERPARAMETERS.ALPHA,
            weight_decay=CONFIG.HYPERPARAMETERS.DECAY
        )
        criterion = nn.SmoothL1Loss()

        # Initialize Trainer
        trainer = RegressionTorchTrainer(model, optimizer, criterion, CONFIG.HYPERPARAMETERS.ACCELERATOR)
        # Start Training
        trainer.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=CONFIG.HYPERPARAMETERS.EPOCHS,
            model_save_path=CONFIG.FILEPATHS.SAVED_MODEL,
        )


if __name__ == "__main__":
    main()
