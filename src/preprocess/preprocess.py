from torchvision import transforms, datasets
import argparse
from utils import utils


def run():
    parser = argparse.ArgumentParser(description='Preprocess pokemon data')
    parser.add_argument('--root', type=str,
                        default='data/raw', help='Dataset root')
    parser.add_argument('--output', type=str, default='data/preprocessed',
                        help='Output directory of preprocessed data')
    dataset_root, output_dir = parser.parse_args().root, parser.parse_args().output
    dataset = preprocess(dataset_root)
    utils.save_dataset(dataset, output_dir, raw_dir=dataset_root)


def preprocess(dataset_root):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(to_rgba),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    return dataset


def to_rgba(x):
    if x.shape[0] == 4:
        return transforms.functional.to_pil_image(x).convert('RGBA')
    else:
        return transforms.functional.to_pil_image(x)


if __name__ == '__main__':
    run()
