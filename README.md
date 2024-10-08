# REDUCR: ROBUST DATA DOWNSAMPLING USING CLASS PRIORITY REWEIGHTING
| **[Abstract](#abstract)**
| **[Installation](#installation)**
| **[Codebase](#codebase)** |

[![arXiv](https://img.shields.io/badge/arXiv-2106.02584-b31b1b.svg)](https://arxiv.org/abs/2206.07137)
[![Python 3.8](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.9-red.svg)](https://shields.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

Here we present the code for our paper [REDUCR: Robust data-downsampling using class priority reweighting](https://arxiv.org/abs/2312.00486). Our additions are implemented on top of the code provided for [Mindermann et al. 2022](https://github.com/OATML/RHO-Loss).

## Abstract
Modern machine learning models are becoming increasingly expensive to train for real-world image and text classification tasks, where massive web-scale data is collected in a streaming fashion. To reduce the training cost, online batch selection techniques have been developed to choose the most informative datapoints. However, these techniques can suffer from poor worst-class generalization performance due to class imbalance and distributional shifts. This work introduces REDUCR, a robust and efficient data downsampling method that uses class priority reweighting. REDUCR reduces the training data while preserving worst-class generalization performance. REDUCR assigns priority weights to datapoints in a class-aware manner using an online learning algorithm. We demonstrate the data efficiency and robust performance of REDUCR on vision and text classification tasks. On web-scraped datasets with imbalanced class distributions, REDUCR achieves significant test accuracy boosts for the worst-performing class (but also on average), surpassing state-of-the-art methods by around 14%.

## Installation
We use python venv and install packages via pip:

```python3 -m pip install -r requirements.txt```

## Codebase
The codebase contains the functionality for all the experiments in the paper. The code uses PyTorch Lightning, Hydra for config file management, and Weights & Biases for logging. 

### Selection method 
REDUCR is implemented in ```src/curricula/selection_methods.py``` as the class ```robust_reducible_loss_selection```.

### Amortised class irreducible loss model training
Start with ```run_irreducible.py```(which then calls ```src/train_irreducible.py```). The base config file is ```configs/irreducible_training.yaml```.

### Target model training
Start with ```run.py```(which then calls ```src/train.py```). The base config file is ```configs/config.yaml```. A key file is ```src//models/MultiModels.py```---this is the LightningModule that handles the training loop incl. batch selection. 

### More about the code
The datamodules are implemented in ```src/datamodules/datamodules.py```, the individual datasets in ```src/datamodules/dataset/sequence_datasets```. If you want to add your own dataset, note that ```__getitem__()``` needs to return the tuple ```(index, input, target, target)```, where ```index``` is the index of the datapoint with respect to the overall dataset (this is required so that we can match the irreducible losses to the correct datapoints), and the second ```target``` is the labels for the robust setting, we have implemented it in this way such that it can be generalised to group robust settings. The experiment configs from [Mindermann et al. 2022](https://github.com/OATML/RHO-Loss) have been moved to ```configs/experiment_old``` and the REDUCR experiment config files can be found in `configs/experiment`. Before running experiments the ```datamodule.data_dir```, ```model_io.home_dir``` and ```logger``` config files should be update with relevent local file paths and wandb details etc...

## Reproducibility
This repo can be used to reproduce all the experiments in the paper. Check out ```configs/experiment``` for some example experiment configs. The experiment files for the main results are: 
* CIFAR-10: ```cifar10_resnet18_irred_weights.yaml``` and ```cifar10_resnet18_robust.yaml```
* CINIC-10: ```cinic10_resnet18_irred_weights.yaml``` and ```cinic10_resnet18_robust.yaml```
* Clothing-1M: ```clothing1m_irred_weights.yaml``` and ```clothing1m_main_robust.yaml```
* MNLI: ```mnli_irred_weights.yaml``` and ```mnli_robust.yaml```

The mnli_irred_weights models must be trained for each $c \in C$, this is done by adjusting the ```gradient_weighted_class``` parameter. 
