import argparse
from utils.utils import get_model, get_loader, save_loss_acc
import torch
import torch.nn as nn


def run():
    dataset_path, model_path = get_args()
    model = get_model(model_path)
    loader = get_loader(dataset_path)
    loss, accuracy = evaluate(model, loader)
    save_loss_acc(loss, accuracy, mode='test')


def evaluate(model, loader):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        loss = running_loss / len(loader.dataset)
        accuracy = running_corrects.double() / len(loader.dataset)
        return loss, accuracy


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--dataset_path', type=str,
                        default='data/splits/test', help='Dataset root')
    parser.add_argument('--model_path', type=str, default='saved_models/model1.pt',
                        help='Model path')
    return parser.parse_args().dataset_path, parser.parse_args().model_path


if __name__ == "main":
    run()
