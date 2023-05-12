import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(PARENT_DIR)

import argparse
from utils.dvc_utils import get_model
from PIL import Image
from preprocess import preprocess
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run():
    images_dir, model_path = get_args()
    images_dir = os.path.join(PARENT_DIR, images_dir)   
    model_path = os.path.join(PARENT_DIR, model_path)
    predictions = []
    model = get_model(model_path)
    model.eval()
    for filename in os.listdir(images_dir):
        image = Image.open(os.path.join(images_dir, filename))
        image = preprocess(image)
        predictions.append(model(image))
    logger.info('The model\'s predictions : ', predictions)


def get_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--images-path', type=str,
                        default='data/inference', help='Images root')
    parser.add_argument('--model-path', type=str, default='saved_models/model.pt',
                        help='Model path')
    args = parser.parse_args()
    return args.images_path, args.model_path


if __name__ == "__main__":
    run()
