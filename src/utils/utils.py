import os
import shutil
import torch
import warnings
from torchvision import datasets, transforms
from torchvision.utils import save_image, Image as Img
import logging
from PIL import Image
import numpy as np
import mlflow
import mlflow.pytorch

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def save_dataset(dataset, output_dir):
    labels = dataset.classes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the transformation
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB')
    ])

    for i, (image, label_idx) in enumerate(dataset):
        label = labels[label_idx]
        image_np = image.mul(255).byte().numpy()
        # transform to the format expected by PIL
        image_pil = Image.fromarray(np.transpose(image_np, (1, 2, 0)))
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        filename = f"{i}.png"
        filepath = os.path.join(label_dir, filename)
        image = transform(image)
        image_pil.save(filepath)


# def save_dataset(dataset, output_dir):
#     labels = dataset.classes
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for i, (image, label_idx) in enumerate(dataset):
#         # image = Image.fromarray()
#         label = labels[label_idx]
#         label_dir = os.path.join(output_dir, str(label))
#         if not os.path.exists(label_dir):
#             os.makedirs(label_dir)
#         filename = f"{i}.png"
#         filepath = os.path.join(label_dir, filename)
#         # Denormalize the image
#         # print(image.shape)
#         # image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
#         # image = torch.clamp(image, 0, 1)
#         # print(image.shape)
#         # transform = transforms.Compose([
#         #     transforms.ToPILImage(mode='RGB')
#         # ])
#         # image = transform(image)
#         # image.save(filepath)
#         save_image(image, filepath, format='png')
#         image = Img.open(filepath, formats=['png'])
#         transform = transforms.Compose([transforms.ToTensor()])
#         image = transform(image)
#         mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
#         image = image * std + mean
#         # transform = transforms.Compose([
#         #     transforms.ToPILImage(mode='RGB')
#         # ])
#         print(filepath)
#         # image = transform(image)
#         save_image(image, filepath, format='png')
#         break

def save_files(dir, files, dataset_dir):
    """
    Saves a list of files from the dataset directory to a specified directory.

    Args:
        dir (str): The directory to save the files to.
        files (List[str]): A list of file names to save.
        dataset_dir (str): The directory where the files are located.

    Returns:
        None.

    Example:
        save_files("data/splits/train/Fearow", ["34.png", "6.png"], "data/preprocessed")

    This will save the "34.png" and "6.png" files from the "data/preprocessed" directory to the "data/splits/train/Fearow" directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        src_file = os.path.join(dataset_dir, file)
        dst_file = os.path.join(dir, file)
        shutil.copy(src_file, dst_file)


def get_loader(path, batch_size=16, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader, len(set(dataset.targets))


def get_image(image_path):
    image = Image.open(image_path)
    print(type(image))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)


def get_model(model_path):
    model = torch.load(model_path)
    return model


def save_image(image, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_path = os.path.join(image_path, 'visualization.png')
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB')
    ])
    image = transform(image)
    image.save(image_path)


# def save_model(model, model_path):
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     model_name = 'model.pt'
#     torch.save(model, model_path + '/' + model_name)


# def save_loss_acc(loss, accuracy, mode='val'):
#     with open(mode + "_loss_acc.txt", "w") as f:
#         f.write(f"Validation Loss: {loss}\nValidation Accuracy: {accuracy}")
