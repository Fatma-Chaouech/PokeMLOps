import sys
sys.path.append('src')
from typing import List
from mlflow.tracking import MlflowClient
import mlflow
import os
from typing import Dict



def setup_mlflow(experiment_name):
    # initialize the MLFlow client
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    # create a new tracing client that uses the MLFlow client and the OpenTelemetry tracer provider
    mlflow_client = MlflowClient()
    return mlflow_client


def log_acc_loss(loss, accuracy, epoch = 0):
    mlflow.log_metric("loss", loss, epoch)
    mlflow.log_metric("accuracy", accuracy, epoch)


def log_metrics(metrics: Dict[str, float], epoch = 0):
    mlflow.log_metrics(metrics, step=epoch)


def log_params(num_epochs, batch_size, learning_rate):
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)


def log_model(model, model_dir):
    mlflow.pytorch.log_model(model, model_dir)


def log_artifact(artifact, name):
    mlflow.log_artifact(artifact, artifact_path=name)


def register_model(run_id, model_name, model_description):

    load_id = os.path.join("runs:/" + run_id, "model")
    register_id = os.path.join("runs:/" + run_id, model_name)
    model = mlflow.pyfunc.load_model(load_id)

    # Register the best model with a unique name
    mlflow.pyfunc.log_model(model, model_name)
    mlflow.register_model(register_id, model_name,
                          description=model_description)
