import sys
sys.path.append('src')
from utils.dvc_utils import push_model
from utils.mlflow_utils import register_model
import mlflow.pyfunc
import mlflow
import argparse


def run():
    run_id, model_name, model_description, push_to_dvc = get_args()
    register_model(run_id, model_name, model_description)
    # if push_to_dvc:
    #     push_model(run_id, model_name)


def get_args():
    parser = argparse.ArgumentParser(
        description="Register a model with MLflow.")
    parser.add_argument('run_id', type=int, required=True,
                        help="The id of the run to register.")
    parser.add_argument('model_name', type=str, required=True,
                        help="The chosen registration name.")
    parser.add_argument('--model_description', type=str, required=False,
                        default="", help="Optional description for the registered model.")
    parser.add_argument('--push_to_dvc', type=bool, required=False,
                        default=True, help="Optional description for the registered model.")
    args = parser.parse_args()
    return args.run_id, args.model_name, args.model_description, args.push_to_dvc


if __name__ == "__main__":
    run()
