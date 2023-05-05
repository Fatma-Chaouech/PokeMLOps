import mlflow
from mlflow.tracking import MlflowClient
from typing import List


def setup_mlflow(experiment_name):
    # initialize the MLFlow client
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    # create a new tracing client that uses the MLFlow client and the OpenTelemetry tracer provider
    mlflow_client = MlflowClient()
    return mlflow_client


def log_acc_loss(loss, accuracy, epoch, mode='train'):
    mlflow.log_metric("loss", loss, epoch, mode)
    mlflow.log_metric("accuracy", accuracy, epoch, mode)


def log_metric(name, values: List[float], info: str = None):
    mlflow.log_metric(name, *values, info)


def log_params(num_epochs, batch_size, learning_rate):
    mlflow.log_metric("num_epochs", num_epochs)
    mlflow.log_metric("batch_size", batch_size)
    mlflow.log_metric("learning_rate", learning_rate)


def log_model(model, model_dir):
    mlflow.pytorch.log_model(model, model_dir)


def log_artifact(artifact, name):
    mlflow.log_artifact(artifact, artifact_path=name)

