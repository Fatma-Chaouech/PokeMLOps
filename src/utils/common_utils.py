import os
import torch
import warnings
from torchvision import datasets, transforms
import logging
from PIL import Image


warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



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
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)


def save_image(image, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image_path = os.path.join(image_path, 'visualization.png')
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB')
    ])
    image = transform(image)
    image.save(image_path)
