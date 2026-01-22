<h1 align="center">AdaFM</h1>

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

</div>

## Abstract

Here for the abstract of your project. This section should provide a brief overview of what the project is about, its purpose, and its key features.

## Installation and Usage

### 1. Prerequisites

Ensure you have Python 3.13+ and `conda` installed on your system.

### 2. Setup Steps

```bash
# 1. Clone this repository
git clone https://github.com/PlamephiaJ/AdaFM
cd AdaFM

# 2. Create a virtual environment using conda
conda create --name AdaFM python=3.13

# 3. Activate the virtual environment on Ubuntu 24.04
conda activate AdaFM

# 4. Install project dependencies
pip install -r requirements.txt
```


### 3. Running an Experiment

Single run (no hyperparameter search):

```bash
python main.py optuna.enabled=false
```

Optuna hyperparameter search (uses settings in configs/base.yaml):

```bash
python main.py optuna.enabled=true optuna.n_trials=20
```

You can edit the Optuna search space in configs/base.yaml under the `optuna` section.

ImageNet (128x128) defaults:

- Dataset config: configs/datasets/imagenet.yaml
- Backbone: configs/models/backbone/wgan-gp-in-128.yaml

Expected ImageNet folder layout:

- <dataroot>/imagenet/train
- <dataroot>/imagenet/val

Multi-machine search (shared PostgreSQL storage):

1) Set `optuna.storage` in configs/base.yaml, for example:
	postgresql+psycopg2://user:password@host:5432/optuna_db
2) Run the same command on each machine:

```bash
python main.py optuna.enabled=true optuna.storage=postgresql+psycopg2://user:password@host:5432/optuna_db
```


## Expected Results

(Optional) You can showcase example plots or key results from your project runs here, such as:
* A comparison of model accuracy under different aggregation rules.
* The loss curve convergence with and without Byzantine attacks.

![Model Accuracy Comparison](placeholder_accuracy_plot.png)

## 📄 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.