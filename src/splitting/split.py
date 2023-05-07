import sys
sys.path.append('src')
import argparse
from sklearn.model_selection import train_test_split
import os
from utils.dvc_utils import save_files


def run():
    dataset_dir, output_dir, train_perc, val_perc, random_state = get_args()
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        train_sub_dir = os.path.join(train_dir, folder)
        val_sub_dir = os.path.join(val_dir, folder)
        test_sub_dir = os.path.join(test_dir, folder)
        train_files, val_files, test_files = generate_splits(
            folder_path, train_perc, val_perc, random_state)
        save_files(train_sub_dir, train_files, folder_path)
        save_files(val_sub_dir, val_files, folder_path)
        save_files(test_sub_dir, test_files, folder_path)


def generate_splits(folder_path, train_perc, val_perc, random_state):
    all_files = os.listdir(folder_path)
    test_perc = 1 - train_perc - val_perc
    train_files, test_files = train_test_split(
        all_files, test_size=test_perc, random_state=random_state)
    train_files, val_files = train_test_split(
        train_files, test_size=val_perc/(train_perc+val_perc), random_state=random_state)
    return train_files, val_files, test_files


def get_args():
    parser = argparse.ArgumentParser(description='Split pokemon data')
    parser.add_argument('--root', type=str,
                        default='data/preprocessed', help='Dataset root')
    parser.add_argument('--output', type=str, default='data/splits',
                        help='Output directory of the splits')
    parser.add_argument('--train_percentage', type=float, default=0.7,
                        help='Train split percentage of the entire dataset')
    parser.add_argument('--val_percentage', type=float, default=0.15,
                        help='Validation split percentage of the entire dataset')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed of the splits')
    args = parser.parse_args()
    return args.root, args.output, args.train_percentage, args.val_percentage, args.random_state


if __name__ == '__main__':
    run()
