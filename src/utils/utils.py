import os
import shutil
import torch
import warnings
from torchvision import datasets
import logging

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_dataset(dataset, output_dir, raw_dir):
    labels = get_labels(raw_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (image, label_idx) in enumerate(dataset):
        label = labels[label_idx]
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        filename = f"{i}.pt"
        filepath = os.path.join(label_dir, filename)
        torch.save(image, filepath)


def get_labels(root):
    """
    Gets a list of class labels from the subdirectories in the specified root directory.

    Args:
        root (str): The root directory containing the subdirectories of class labels.

    Returns:
        labels (list): A list of class labels.
    """
    labels = []
    for subdir in os.listdir(root):
        if os.path.isdir(os.path.join(root, subdir)):
            labels.append(subdir)
    return labels


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
        save_files("data/splits/train/Fearow", ["34.pt", "6.pt"], "data/preprocessed")

    This will save the "34.pt" and "6.pt" files from the "data/preprocessed" directory to the "data/splits/train/Fearow" directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        src_file = os.path.join(dataset_dir, file)
        dst_file = os.path.join(dir, file)
        shutil.copy(src_file, dst_file)
    logger.info('Finished with ', dir)


def get_loader(path, batch_size=64, shuffle=True):
    print(path)
    dataset = datasets.ImageFolder(path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return loader


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_loss_acc(loss, accuracy):
    with open("val_loss_acc.txt", "w") as f:
        f.write(f"Validation Loss: {loss}\nValidation Accuracy: {accuracy}")
