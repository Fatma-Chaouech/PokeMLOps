# PokeMLOps

PokeMLOps's main objective is to provide a comprehensive and user-friendly MLOps platform by integrating powerful open-source tools that enable easy management, deployment, and evaluation of machine learning models. 

### Description
This project delivers an efficient MLOps solution for managing and deploying ML models with ease. The model's performance and robustness is taken into consideration to ensure that it can handle noise, data transformations, and data drift. In addition to this, the project has a monitoring system that tracks various metrics using Open Telemetry. The project also uses MLFlow to track experiments, manage models, and create projects.

### Setup
1. Clone the repository 
```
git clone https://github.com/Fatma-Chaouech/PokeMLOps.git
``` 
2. Install conda
3. Create the environment
```
conda env create -f environment.yml
``` 
4. Activate the environment
```
conda activate pokenv
```
5. Pull the dataset
```
dvc pull
```
6. set a PYTHONPATH environment variable
```
export PYTHONPATH=$PWD/src:$PYTHONPATH
```
### Usage
1. Navigate to the project directory
```
cd PokeMLOps
```
2. Preprocess the dataset
```
python3 src/preprocessing/preprocess.py [--root ROOT] [--output OUTPUT]
```
optional arguments:
  --root ROOT           Dataset root directory (default: data/raw)
  --output OUTPUT       Directory to save preprocessed data (default: data/preprocessed)

3. Split the dataset
```
python3 src/splitting/split.py [--root ROOT] [--output OUTPUT] [--train_percentage TRAIN_PERCENTAGE] [--val_percentage VAL_PERCENTAGE] [--random_state RANDOM_STATE]
```
optional arguments:
  --root ROOT                Dataset root directory (default: data/preprocessed)
  --output OUTPUT            Output directory of the splits (default: data/splits)
  --train_percentage TRAIN_PERCENTAGE    Train split percentage of the entire dataset (default: 0.7)
  --val_percentage VAL_PERCENTAGE        Validation split percentage of the entire dataset (default: 0.15)
  --random_state RANDOM_STATE           Random seed of the splits (default: 42)
4. Train the model
```
python3 src/training/train.py [--train_path TRAIN_PATH] [--val_path VAL_PATH] [--model_path MODEL_PATH] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
```
optional arguments:
  --train_path TRAIN_PATH           Path to the training data (default: data/splits/train)
  --val_path VAL_PATH               Path to the validation data (default: data/splits/val)
  --model_path MODEL_PATH           Path to save the trained model (default: saved_models)
  --num_epochs NUM_EPOCHS           Number of epochs to train for (default: 20)
  --batch_size BATCH_SIZE           Batch size for training (default: 64)
  --learning_rate LEARNING_RATE     Learning rate for training (default: 0.01)


Note that there are default values for the arguments.
