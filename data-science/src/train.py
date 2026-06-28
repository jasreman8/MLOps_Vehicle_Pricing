# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")

    parser.add_argument("--train_data", type=str, help="Path to train data directory")
    parser.add_argument("--test_data", type=str, help="Path to test data directory")
    parser.add_argument("--model_output", type=str, help="Path to save the trained model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the trees")

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    train_df = pd.read_csv(glob.glob(os.path.join(args.train_data, "*.csv"))[0])
    test_df = pd.read_csv(glob.glob(os.path.join(args.test_data, "*.csv"))[0])

    X_train = train_df.drop(columns=["price"])
    y_train = train_df["price"]
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("MSE", mse)

    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(model, args.model_output)


if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
