import sys
sys.path.append('src')
import argparse
from utils.opentelemetry_utils import get_telemetry_args
from utils.common_utils import get_loader
from utils.dvc_utils import get_model
from utils.mlflow_utils import log_acc_loss, log_artifact
import torch
from torchmetrics import Accuracy
import torch.nn as nn
from sklearn.metrics import classification_report


def run():
    dataset_path, model_path = get_args()
    model = get_model(model_path)
    loader, _ = get_loader(dataset_path)
    loss, accuracy, report = evaluate(model, loader)
    log_acc_loss(loss, accuracy)
    log_artifact(report, name='classification_report')


def evaluate(model, loader):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    accuracy = Accuracy(task="multiclass", num_classes=model.num_classes)
    running_corrects = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += accuracy(preds, labels.data)

            # Append true and predicted labels for classification report
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        loss = running_loss / len(loader.dataset)
        acc = running_corrects.double() / len(loader.dataset)

        report = classification_report(y_true, y_pred)

        return loss, acc, report


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    telemetry_args = get_telemetry_args(parser)
    parser.add_argument('--dataset_path', type=str,
                        default='data/splits/test', help='Dataset root')
    parser.add_argument('--model_path', type=str, default='saved_models/model.pt',
                        help='Model path')
    args = parser.parse_args()
    return *telemetry_args, args.dataset_path, args.model_path


if __name__ == "__main__":
    run()
