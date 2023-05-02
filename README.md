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
Optional arguments:
* --root : Dataset root directory (default: data/raw)
* --output : Directory to save preprocessed data (default: data/preprocessed)

3. Split the dataset
```
python3 src/splitting/split.py [--root ROOT] [--output OUTPUT] [--train_percentage TRAIN\_PERCENTAGE] [--val_percentage VAL\_PERCENTAGE] [--random\_state RANDOM\_STATE]
```

Optional arguments:

*  --root         :     Dataset root directory (default: data/preprocessed) 

* --output    : Output directory of the splits (default: data/splits) 

* --train\_percentage :  Train split percentage of the entire dataset (default: 0.7) 

* --val\_percentage   :  Validation split percentage of the entire dataset (default: 0.15) 

* --random\_state  : Random seed of the splits (default: 42)

4. Train the model
```
python3 src/training/train.py [--train\_path TRAIN\_PATH] [--val\_path VAL\_PATH] [--model\_path MODEL\_PATH] [--num\_epochs NUM\_EPOCHS] [--batch\_size BATCH\_SIZE] [--learning\_rate LEARNING\_RATE]
```

Optional arguments:

* --train\_path :Path to the training data (default: data/splits/train)

* --val\_path : Path to the validation data (default: data/splits/val)

* --model\_path :Path to save the trained model (default: saved\_models)

* --num\_epochs:  Number of epochs to train for (default: 20)

* --batch\_size  :Batch size for training (default: 64)

* --learning\_rate : Learning rate for training (default: 0.01)


Note that there are default values for the arguments.
