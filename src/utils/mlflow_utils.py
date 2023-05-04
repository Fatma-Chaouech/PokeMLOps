import mlflow

def log_metrics(loss, accuracy, epoch, mode='train'):
    mlflow.log_metric("loss", loss, epoch, mode)
    mlflow.log_metric("accuracy", accuracy, epoch, mode)


def log_params(num_epochs, batch_size, learning_rate):
    mlflow.log_metric("num_epochs", num_epochs)
    mlflow.log_metric("batch_size", batch_size)
    mlflow.log_metric("learning_rate", learning_rate)


def log_model(model, model_dir):
    mlflow.pytorch.log_model(model, model_dir)


def log_artifact(artifact, name):
    mlflow.log_artifact(artifact, artifact_path=name)
