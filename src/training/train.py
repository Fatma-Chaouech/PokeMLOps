import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(PARENT_DIR)

import argparse
from torchmetrics import Accuracy, AUROC, F1Score, Precision, Recall
import torch
import torch.nn as nn
from src.models.model import PokeModel
from src.utils.opentelemetry_utils import get_telemetry_args, setup_telemetry
from src.utils.common_utils import get_loader
from src.utils.mlflow_utils import log_acc_loss, log_params, setup_mlflow, log_model, log_metrics
import mlflow
import warnings
import logging


warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def run():
    experiment_name, train_path, val_path, model_path, model_name, num_epochs, batch_size, learning_rate = get_args()
    train_path = os.path.join(PARENT_DIR, train_path)
    val_path = os.path.join(PARENT_DIR, val_path)
    # model_path = os.path.join(PARENT_DIR, model_path)
    tracer = setup_telemetry()
    client = setup_mlflow(experiment_name)
    with mlflow.start_run():
        with tracer.start_as_current_span("training") as train_tracer:
            log_params(num_epochs, batch_size, learning_rate)
            trainloader, num_classes = get_loader(train_path, batch_size)
            valloader, _ = get_loader(val_path, batch_size)
            classifier = PokeModel(num_classes)
            trainer = PokeTrainer(classifier, trainloader,
                                  valloader, learning_rate)
            classifier.model.requires_grad = True
            trainer.train(num_epochs, model_path, model_name, train_tracer)


class PokeTrainer():
    def __init__(self, model, trainloader, valloader, learning_rate=0.001, momentum=0.9, step_size=7, gamma=0.1):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print('training on...', self.device)
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum)
        num_classes = self.model.num_classes
        metrics = get_metrics(num_classes)
        self.accuracy = metrics[2].to(self.device)
        # self.auroc = metrics[4].to(self.device)
        self.f1 = metrics[3].to(self.device)
        self.precision = metrics[0].to(self.device)
        self.recall = metrics[1].to(self.device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

    def train(self, num_epochs, model_path, model_name, train_tracer):
        best_acc = 0.0
        loss = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            logger.info(f"Epoch {epoch + 1} / {num_epochs}")

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += self.accuracy(preds, labels.data)
            self.scheduler.step()
            epoch_loss = running_loss / len(self.trainloader.dataset)
            epoch_acc = running_corrects.double() / len(self.trainloader.dataset)
            log_acc_loss(epoch_loss, epoch_acc, epoch)
            self.model.eval()
            val_loss, val_acc, val_precision, val_recall, val_f1 = self._evaluate()
            if val_acc > best_acc:
                best_acc = val_acc
                loss = val_loss
                log_metrics({'precision': val_precision, 'recall': val_recall, 'f1': val_f1}, epoch)
                log_acc_loss(loss, val_acc, epoch)
                log_model(self.model, model_path)

    def _evaluate(self):
        with torch.no_grad():
            val_running_loss = 0.0
            val_running_corrects = 0
            val_predcision = 0
            val_recall = 0
            # val_auroc = 0
            val_f1 = 0  
            for val_inputs, val_labels in self.valloader:
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)

                val_outputs = self.model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_loss = self.criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_inputs.size(0)
                val_running_corrects += self.accuracy(
                    val_preds, val_labels.data)
                val_predcision += self.precision(val_preds, val_labels.data)
                val_recall += self.recall(val_preds, val_labels.data)
                # val_auroc += self.auroc(val_preds, val_labels.data)
                val_f1 += self.f1(val_preds, val_labels.data)

            val_loss = val_running_loss / len(self.valloader.dataset)
            val_acc = val_running_corrects.double() / len(self.valloader.dataset)
            val_precision = val_predcision / len(self.valloader.dataset)
            val_recall = val_recall / len(self.valloader.dataset)
            # val_auroc = val_auroc / len(self.valloader.dataset)
            val_f1 = val_f1 / len(self.valloader.dataset)
        return val_loss, val_acc, val_precision, val_recall, val_f1


def get_metrics(num_classes):
    # auroc = AUROC(task="multiclass", num_classes=num_classes)
    recall = Recall(task="multiclass", num_classes=num_classes)
    precision = Precision(task="multiclass", num_classes=num_classes)
    f1 = F1Score(task="multiclass", num_classes=num_classes)
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)
    return precision, recall, accuracy, f1#, auroc

def get_args():
    parser = argparse.ArgumentParser(description='Train pokemon classifier')
    parser = get_telemetry_args(parser)
    parser.add_argument('--train-path', type=str,
                        default='data/preprocessed/train', help='Training data root')
    parser.add_argument('--val-path', type=str,
                        default='data/preprocessed/val', help='Validation data root')
    parser.add_argument('--model-path', type=str,
                        default='saved_models', help='Saved model path')
    parser.add_argument('--model-name', type=str,
                        default='saved_models', help='Saved model name')
    parser.add_argument('--num-epochs', type=int,
                        default=20, help='Number of epochs of the training')
    parser.add_argument('--batch-size', type=int,
                        default=8, help='Batch size of the training')
    parser.add_argument('--learning-rate', type=float,
                        default=0.001, help='Learning rate of the training')
    args = parser.parse_args()
    return args.experiment_name, args.train_path, args.val_path, args.model_path, args.model_name, args.num_epochs, args.batch_size, args.learning_rate


if __name__ == "__main__":
    run()
