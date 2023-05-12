# PokeMLOps
![pokemon](docs/pokemon.png)
PokeMLOps's main objective is to provide a comprehensive and user-friendly MLOps platform by integrating powerful open-source tools that enable easy management, deployment, and evaluation of machine learning models. 

### Description
This project delivers an efficient MLOps solution for managing and deploying ML models with ease, applied to a [Pokemon Generation One](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) classifier. The model's performance and robustness is taken into consideration to ensure that it can handle noise, data transformations, and data drift. In addition to this, the project has a monitoring system that tracks various metrics using Open Telemetry. The project also uses MLFlow to track experiments, manage models, and create projects.

### Setup
1. Clone the repository 
```
git clone https://github.com/Fatma-Chaouech/PokeMLOps.git
``` 
2. Install conda
3. Navigate to the project directory
```
cd PokeMLOps
```
4. Create and activate the environment
```
conda create --name pokenv
conda activate pokenv
conda env update --file environment.yml
``` 
5. Pull the dataset
```
dvc pull
```
### Usage
1. Open MLFlow UI to track the experiments
```
mlflow ui
```
2. Visualize the pipeline
```
dvc dag 
```
3. Run the pipeline
```
dvc repro
```
>If you wan to run a specific stage, run the same command followed by the name of the stage:
`dvc repro STAGE` 

**There are default values for the arguments.**
