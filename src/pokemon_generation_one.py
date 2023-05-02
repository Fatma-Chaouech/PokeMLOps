import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import numpy as np
import time
import copy
from torch.utils.data import random_split
# %matplotlib inline


DATASET_ROOT = '/data/raw'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: transforms.functional.to_pil_image(x).convert(
        'RGBA') if x.shape[0] == 4 else transforms.functional.to_pil_image(x)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for the training and testing sets
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2)

# get the labels
labels = []
for _, dirs, _ in os.walk(DATASET_ROOT):
    labels.extend(dirs)
labels = [label for label in labels if label != '']


def imshow(img, ax, transform):
    img = transform(img)
    ax.imshow(img)


def show_n_batches(loader, n, figsize=(20, 20), ncols=4):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.numpy().transpose((1, 2, 0))),
        transforms.Lambda(lambda x: np.clip(
            x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1))
    ])
    for images, targets in loader:
        fig, axes = plt.subplots(figsize=figsize, ncols=ncols)
        for idx in range(ncols):
            if n == 0:
                break
            imshow(images[idx], ax=axes[idx], transform=transform)
            n -= 1


show_n_batches(trainloader, n=4)


def imshow(img, ax):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.numpy().transpose((1, 2, 0))),
        transforms.Lambda(lambda x: np.clip(
            x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1))
    ])
    img = transform(img)
    ax.imshow(img)


def show_n_batches(loader, n, figsize=(20, 20), ncols=4):

    while n:
        images, targets = next(iter(loader))
        fig, axes = plt.subplots(figsize=figsize, ncols=ncols)
        for idx in range(ncols):
            ax = axes[idx]
            print(labels[targets[idx]])
            imshow(images[idx], ax=ax)
        n -= 1


show_n_batches(trainloader, n=4)


def train_model(model, trainloader, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains a PyTorch model for a given number of epochs using the specified
    loss function, optimizer, and learning rate scheduler.

    Args:
        model: PyTorch model to be trained
        trainloader: PyTorch DataLoader for the training data
        criterion: PyTorch loss function
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        num_epochs: Number of epochs to train the model for (default: 25)

    Returns:
        model: Trained PyTorch model
        best_acc: Best validation accuracy achieved during training
    """

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    since = time.time()

    # Initialize best accuracy and model weights
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Set model to training mode
    model.train()

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over the training data
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Enable gradient calculation
            with torch.enable_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # Update learning rate scheduler
        scheduler.step()

        # Compute epoch loss and accuracy
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Update best accuracy and model weights
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc


model = torch.hub.load('pytorch/vision:v0.10.0',
                       'vgg11', weights=True).to(device)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier.requires_grad = True
num_ftrs = model.classifier[6].in_features
model.fc = torch.nn.Linear(num_ftrs, len(labels))
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=7, gamma=0.1)

model = train_model(model, trainloader, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=20)

for param in model.features.parameters():
    param.requires_grad = True

model = train_model(model, trainloader, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=4)


def eval_model(model, testloader, criterion):
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    loss = running_loss / len(trainloader.dataset)
    accuracy = running_corrects.double() / len(trainloader.dataset)
    print('---------------------------------------------')
    print(f"Loss on test: {loss:.4f} Acc: {accuracy:.4f}")
    print('---------------------------------------------')


eval_model(model, testloader, criterion)
