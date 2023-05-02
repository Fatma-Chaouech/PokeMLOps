import argparse
import torch
import torch.nn as nn
from models.model import PokeModel
from utils.utils import get_loader, save_model, save_loss_acc


def run():
    train_path, val_path, model_path, num_epochs, batch_size, learning_rate = get_args()
    trainloader, num_classes = get_loader(train_path, batch_size)
    valloader, _ = get_loader(val_path, batch_size)
    classifier = PokeModel(num_classes)
    trainer = PokeTrainer(classifier, trainloader, valloader, learning_rate)
    classifier.model.requires_grad = True
    classifier, val_loss, val_accuracy = trainer.train(num_epochs)
    save_model(classifier, model_path)
    save_loss_acc(val_loss, val_accuracy)


class PokeTrainer():
    def __init__(self, model, trainloader, valloader, learning_rate=0.001, momentum=0.9, step_size=7, gamma=0.1):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

    def train(self, num_epochs):
        self.model.train()
        best_acc = 0.0
        loss = 0
        best_model_wts = self.model.state_dict()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 10)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            self.scheduler.step()

            epoch_loss = running_loss / len(self.trainloader.dataset)
            epoch_acc = running_corrects.double() / len(self.trainloader.dataset)

            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            val_loss, val_acc = self._evaluate()

            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            print()

            if val_acc > best_acc:
                best_acc = val_acc
                loss = val_loss
                best_model_wts = self.model.state_dict()

        self.model.load_state_dict(best_model_wts)
        return self.model, loss, best_acc

    def _evaluate(self):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            loss = running_loss / len(self.trainloader.dataset)
            accuracy = running_corrects.double() / len(self.trainloader.dataset)
            return loss, accuracy


def get_args():
    parser = argparse.ArgumentParser(description='Train pokemon classifier')
    parser.add_argument('--train_path', type=str,
                        default='data/splits/train', help='Training data root')
    parser.add_argument('--val_path', type=str,
                        default='data/splits/val', help='Validation data root')
    parser.add_argument('--model_path', type=str,
                        default='saved_models', help='Saved model path')
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='Number of epochs of the training')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='Batch size of the training')
    parser.add_argument('--learning_rate', type=float,
                        default=0.01, help='Learning rate of the training')
    args = parser.parse_args()
    return args.train_path, args.val_path, args.model_path, args.num_epochs, args.batch_size, args.learning_rate


if __name__ == "__main__":
    run()
