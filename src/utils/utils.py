import os
import shutil
import torch
import warnings
from torchvision import datasets, transforms
import logging
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_loss_acc(loss, accuracy):
    with open("val_loss_acc.txt", "w") as f:
        f.write(f"Validation Loss: {loss}\nValidation Accuracy: {accuracy}")
