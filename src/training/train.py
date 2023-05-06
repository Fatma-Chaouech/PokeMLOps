import argparse
from torchmetrics import Accuracy
import torch
import torch.nn as nn
from models.model import PokeModel
from opentelemetry import metrics
from utils.opentelemetry_utils import get_telemetry_args, setup_telemetry
from utils.common_utils import get_loader
from utils.mlflow_utils import log_acc_loss, log_params, setup_mlflow, log_model
import mlflow
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def run():
    experiment_name, train_path, val_path, model_path, num_epochs, batch_size, learning_rate = get_args()
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
            trainer.train(num_epochs, model_path, train_tracer)


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
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)
        self.meter = metrics.get_meter(__name__)

    def train(self, num_epochs, model_path, train_tracer):
        best_acc = 0.0
        loss = 0.0
        epoch_counter = self.meter.create_counter(
            "epoch_counter",
            description="The number of passed epochs",
        )
        train_tracer.set_attribute("train.epoch", 0)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_counter.add(1)
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
            log_acc_loss(epoch_loss, epoch_acc, epoch, mode='train')
            self.model.eval()
            val_loss, val_acc = self._evaluate()
            if val_acc > best_acc:
                best_acc = val_acc
                loss = val_loss
                log_acc_loss(loss, val_acc, epoch, mode='val')
                log_model(self.model, model_path)

    def _evaluate(self):
        with torch.no_grad():
            val_running_loss = 0.0
            val_running_corrects = 0
            for val_inputs, val_labels in self.valloader:
                val_inputs = val_inputs.to(self.device)
                val_labels = val_labels.to(self.device)

                val_outputs = self.model(val_inputs)
                _, val_preds = torch.max(val_outputs, 1)
                val_loss = self.criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_inputs.size(0)
                val_running_corrects += self.accuracy(
                    val_preds, val_labels.data)

            val_loss = val_running_loss / len(self.valloader.dataset)
            val_acc = val_running_corrects.double() / len(self.valloader.dataset)
        return val_loss, val_acc


def get_args():
    parser = argparse.ArgumentParser(description='Train pokemon classifier')
    parser = get_telemetry_args(parser)
    parser.add_argument('--train_path', type=str,
                        default='data/splits/train', help='Training data root')
    parser.add_argument('--val_path', type=str,
                        default='data/splits/val', help='Validation data root')
    parser.add_argument('--model_path', type=str,
                        default='saved_models', help='Saved model path')
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='Number of epochs of the training')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='Batch size of the training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate of the training')
    args = parser.parse_args()
    return args.experiment_name, args.train_path, args.val_path, args.model_path, args.num_epochs, args.batch_size, args.learning_rate


if __name__ == "__main__":
    run()
