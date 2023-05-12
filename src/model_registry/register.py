import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(PARENT_DIR)

import argparse
from utils.mlflow_utils import register_model


def run():
    run_id, model_path, model_description = get_args()
    register_model(run_id, model_path, model_description)


def get_args():
    parser = argparse.ArgumentParser(
        description="Register a model with MLflow.")
    parser.add_argument('run-id', type=int, required=True,
                        help="The id of the run to register.")
    parser.add_argument('model-path', type=str, required=True,
                        help="The chosen registration name.")
    parser.add_argument('--model-description', type=str, required=False,
                        default="", help="Optional description for the registered model.")
    args = parser.parse_args()
    return args.run_id, args.model_path, args.model_description


if __name__ == "__main__":
    run()
