import os
import sys
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(PARENT_DIR)

import os
import pytest
from PIL import Image
from src.utils.dvc_utils import save_dataset
from src.preprocessing.preprocess import preprocess

@pytest.fixture
def mock_dataset(tmp_path):
    """
    Create a mock dataset with random images for testing purposes.
    """
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    images_path = dataset_path / "images"
    images_path.mkdir()
    for i in range(10):
        img = Image.new("RGB", (100, 100), color=(i, i, i))
        img.save(images_path / f"{i}.jpg")
    return dataset_path

def test_preprocess(mock_dataset, tmp_path):
    """
    Test the preprocess function.
    """
    output_dir = tmp_path / "preprocessed"
    dataset = preprocess(mock_dataset, 'test')
    assert len(dataset) == 10
    assert dataset[0][0].shape == (3, 224, 224)
    assert dataset[0][1] == 0
    save_dataset(dataset, output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(output_dir)) == 1
    assert os.path.isdir(os.path.join(output_dir, "images"))
    assert len(os.listdir(os.path.join(output_dir, "images"))) == 10