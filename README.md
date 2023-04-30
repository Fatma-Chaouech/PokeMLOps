# PokeMLOps

This is an MLOps project aimed at building a robust Pokemon Gen 1 classifier using various technologies such as MLFlow, DVC, Open Telemetry, Hydra, Pylint and Pytest.

### Description
The project's main objective is to create a Pokemon Gen 1 classifier that can accurately predict the type of Pokemon based on their image. The model's performance and robustness will be evaluated using various metrics to ensure that it can handle noise, data transformations, and data drift. In addition to this, the project will have a monitoring system that tracks various metrics using Open Telemetry. The project will also integrate with MLFlow to track experiments, manage models, and create projects.

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
