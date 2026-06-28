# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import os
import json

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print("Registering ", args.model_name)

    # Load the model from path
    model = mlflow.sklearn.load_model(args.model_path)

    # Log the model in the active MLflow run
    mlflow.sklearn.log_model(model, "random_forest_price_regressor")

    # Register the model in the MLflow registry
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/random_forest_price_regressor"
    registered_model = mlflow.register_model(model_uri, args.model_name)

    # Write model info to output path
    os.makedirs(args.model_info_output_path, exist_ok=True)
    model_info = {
        "id": f"{registered_model.name}/{registered_model.version}",
        "name": registered_model.name,
        "version": registered_model.version
    }
    with open(os.path.join(args.model_info_output_path, "model_info.json"), "w") as f:
        json.dump(model_info, f)


if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
