import torch
import torch.nn as nn
from model import PokeModel


def run():
    model = PokeModel()
    pass



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

            val_loss, val_acc = self.evaluate()

            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            print()

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = self.model.state_dict()

        self.model.load_state_dict(best_model_wts)
        return self.model

    def evaluate(self):
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

            epoch_loss = running_loss / len(self.trainloader.dataset)
            epoch_acc = running_corrects.double() / len(self.trainloader.dataset)
            return epoch_loss, epoch_acc



if __name__ == "__main__":
    run()