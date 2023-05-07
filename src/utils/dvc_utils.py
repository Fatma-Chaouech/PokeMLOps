import sys
sys.path.append('src')
import os
import shutil
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

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


def get_model(model_path):
    model = torch.load(model_path)
    return model


# def save_model(model, model_path):
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     model_name = 'model.pt'
#     torch.save(model, model_path + '/' + model_name)
