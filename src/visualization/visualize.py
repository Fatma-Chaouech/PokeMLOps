import torch
from utils.utils import save_image, get_image
import argparse
import numpy as np
from torchvision import transforms
from PIL import Image


def run():
    image_path, output_path = get_args()
    # image = get_image(image_path)
    # image = denormalize(image)
    # save_image(image, output_path)
    image = Image.open(image_path)
    image = image.convert('RGB')
    # Convert to tensor and denormalize
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image)
    image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    # Convert back to image
    image = transforms.ToPILImage()(image)
    # Save the image
    image.save(output_path + '/example.png')


def denormalize(image):
    image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(3,
             1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    image = torch.clamp(image, 0, 1)
    # image = (image * torch.tensor([0.229, 0.224, 0.225]).reshape(3,
    #          1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    # image = torch.clamp(image, 0, 1)
    return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default='data/preprocessed/Abra/0.png', help='path to input image')
    parser.add_argument('--output_path', type=str,
                        default='data/example', help='path to output image')
    args = parser.parse_args()
    return args.input_path, args.output_path


if __name__ == "__main__":
    run()
