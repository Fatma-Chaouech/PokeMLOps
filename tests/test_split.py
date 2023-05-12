import os
import pytest
from splitting.split import generate_splits

@pytest.fixture
def mock_folder(tmp_path):
    # Create a temporary directory with sample files to use for testing
    class_folders = ['class1', 'class2', 'class3']
    for folder in class_folders:
        os.mkdir(os.path.join(tmp_path, folder))
        for i in range(10):
            with open(os.path.join(tmp_path, folder, f'{folder}_img_{i}.jpg'), 'w') as f:
                f.write('mock data')
    return tmp_path

def test_generate_splits(mock_folder):
    # Test that the function returns train, val, and test splits for each class folder
    train_perc = 0.7
    val_perc = 0.15
    random_state = 42
    class_folders = os.listdir(mock_folder)
    for folder in class_folders:
        folder_path = os.path.join(mock_folder, folder)
        train_files, val_files, test_files = generate_splits(folder_path, train_perc, val_perc, random_state)
        # Check that the total number of files is equal to the original number of files
        assert len(train_files) + len(val_files) + len(test_files) == 10
        # Check that the train, val, and test splits are proportional to the specified percentages
        assert len(test_files) == 2
        assert len(train_files) == 6
        assert len(val_files) == 2
