import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import shutil
import os


def save_dataset(dataset, output_dir):
    labels = dataset.classes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in files:
        src_file = os.path.join(dataset_dir, file)
        dst_file = os.path.join(dir, file)
        shutil.copy(src_file, dst_file)
