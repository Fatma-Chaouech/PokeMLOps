import sys
sys.path.append('src')
import logging
from preprocess import preprocess
from PIL import Image
import os
from utils.dvc_utils import get_model
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run():
    images_dir, model_path = get_args()
    predictions = []
    # to change
    model = get_model(model_path)
    model.eval()
    for filename in os.listdir(images_dir):
        image = Image.open(os.path.join(images_dir, filename))
        image = preprocess(image)
        predictions.append(model(image))
    logger.info('The model\'s predictions : ', predictions)


def get_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--images_path', type=str,
                        default='data/inference', help='Images root')
    parser.add_argument('--model_path', type=str, default='saved_models/model.pt',
                        help='Model path')
    args = parser.parse_args()
    return args.images_path, args.model_path


if __name__ == "__main__":
    run()
