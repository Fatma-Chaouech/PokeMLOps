import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(PARENT_DIR)

from src.utils.dvc_utils import save_dataset
import argparse
from torchvision import transforms, datasets


def run():
    dataset_root, output_dir, type = get_args()
    dataset_root = os.path.join(PARENT_DIR, dataset_root)
    output_dir = os.path.join(PARENT_DIR, output_dir)
    dataset = preprocess(dataset_root, phase=type)
    save_dataset(dataset, output_dir)


def preprocess(dataset_root, phase):
    if phase == 'train':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    return dataset


def to_rgb(x):
    if x.shape[0] == 4:
        return transforms.functional.to_pil_image(x).convert('RGB')
    else:
        return transforms.functional.to_pil_image(x)


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess pokemon data')
    parser.add_argument('phase', type=str, choices=['train', 'val', 'test'],
                        help='Preprocessing phase: train, val or test data')
    parser.add_argument('--input-path', type=str,
                        default='data/splits', help='Dataset root')
    parser.add_argument('--output', type=str, default='data/preprocessed',
                        help='Output directory of preprocessed data')
    args = parser.parse_args()
    root = os.path.join(args.input_path, args.phase)
    output = os.path.join(args.output, args.phase)
    return root, output, args.phase


if __name__ == '__main__':
    run()
