# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:25:41 2023

@author: William
"""

import os
import torch
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def download_datasets(dir_path):
    
    file_paths = []
    
    #find files in dir:
    for file in os.listdir(dir_path):
        if file.endswith(".pt"):
            file_paths.append(os.path.join(dir_path, file))
    
    
    return [torch.load(f) for f in file_paths]

        
def load_run(artifact_name, attribute_name):
    run = wandb.init()
    artifact = run.use_artifact(artifact_name)
    artifact_table = artifact.get(attribute_name)
    my_df = pd.DataFrame(data=artifact_table.data, columns=artifact_table.columns)
    wandb.finish()
    
    return my_df


def plot_diversity_selection(tables, _class, alphas, colors, labels):
    
    fig, axs = plt.subplots(figsize=(10,7))

    for i, table in enumerate(tables):
    
        selected_indices = table['selected_index_counts']
        index_targets = table['target_index_counts']
        axs.hist(selected_indices[index_targets == _class[i]],bins=20,
                 label='Total points: {}, {}'.\
                     format(len(index_targets[index_targets==_class[i]]), labels[i]),
                 alpha=alphas[i],
                 color=colors[i])
        axs.set_title(f'Histogram of the number of times each point in class {_class} was selected during training')
        axs.set_xlabel('number of times point selected')
        axs.set_ylabel('number of points in bin')
    plt.legend()
    
   


#%% Download dataset

artifact_name = 'selected_index_counts_table'

cifar_weighted = load_run( #Run with 46% irred model
    'william_bankes/Robust RHO Parallel Runs/run-5hheggpo-selected_index_counts_table:v174',
    artifact_name)
    
cifar_imbal = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-8ztpugcf-selected_index_counts_table:v174',
    artifact_name)

cifar_sampled = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-z9y1qj5j-selected_index_counts_table:v174',
    artifact_name)
    
cifar_softmax_100 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-7x0gekgc-selected_index_counts_table:v173',
    artifact_name)
cifar_softmax_10 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-nxx97jqc-selected_index_counts_table:v173',
    artifact_name)
cifar_softmax_1 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-6e55k2ew-selected_index_counts_table:v174',
    artifact_name)

cifar_softmax_1000 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-8iin4j0m-selected_index_counts_table:v174',
    artifact_name)

cifar_softmax_1e4 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-iekypah3-selected_index_counts_table:v173',
    artifact_name)

cifar_softmax_1e5 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-x7mhydnp-selected_index_counts_table:v174',
    artifact_name)

cifar_softmax_1e6 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-db7fqymn-selected_index_counts_table:v174',
    artifact_name)

rank_1e4 = load_run(
    r'william_bankes/Robust RHO Parallel Runs/run-z2hclms7-selected_index_counts_table:v173',
    artifact_name)

rank_100 = load_run(
    r'william_bankes/Robust RHO Parallel Runs/run-2s6j9gpa-selected_index_counts_table:v174', #100
    artifact_name)

rank_10 = load_run(
    r'william_bankes/Robust RHO Parallel Runs/run-9uscc2q2-selected_index_counts_table:v174', #10
    artifact_name)

rank_1 = load_run(
    r'william_bankes/Robust RHO Parallel Runs/run-8oudm66t-selected_index_counts_table:v173', #1
    artifact_name)

top_k = load_run(
    r'william_bankes/Robust RHO Parallel Runs/run-qq8aor9p-selected_index_counts_table:v174',
    artifact_name)

#%% Visualise dataset for top k selection methods:

plot_diversity_selection([top_k, cifar_imbal],
                         [3,3],
                         [0.7, 0.7],
                         ['tab:orange', 'tab:blue'],
                         ['weighted rho, class 3 ', 'imbalance'])

#%%

plot_diversity_selection([cifar_softmax_1, cifar_softmax_100, cifar_softmax_1e6, cifar_weighted],
                         [1,1,1,1],
                         [1, 0.7, 0.7, 0.7],
                         ['tab:orange','tab:green', 'tab:red', 'tab:blue'],
                         ['beta = 1', 'beta = 100', 'beta = 1e6', 'imbalanced'])

#%%

plot_diversity_selection([cifar_softmax_100, cifar_softmax_100, cifar_softmax_100],
                         [3,1,9],
                         [1, 0.7, 0.7],
                         ['tab:blue','tab:green','tab:orange'],
                         ['class 3', 'class 1', 'class 9'])

#%%

plot_diversity_selection([cifar_softmax_1, cifar_softmax_1, cifar_softmax_1],
                         [3,1,9],
                         [1, 0.7, 0.7],
                         ['tab:blue','tab:green','tab:orange'],
                         ['class 3', 'class 1', 'class 9'])

#%%

plot_diversity_selection([cifar_softmax_1000, cifar_softmax_1e6],
                         [3,3],
                         [1, 0.7],
                         ['tab:blue','tab:green'],
                         ['softmax 1e3', 'softmax 1e6'])

#%%
plot_diversity_selection([cifar_weighted, cifar_softmax_1e6],
                         [3,3],
                         [1, 0.7],
                         ['tab:blue','tab:green'],
                         ['weighted', 'softmax 1e6'])

#%%

plot_diversity_selection([cifar_softmax_1e4, cifar_softmax_1e4, cifar_softmax_1e4],
                         [3,1,9],
                         [1, 0.7, 0.7],
                         ['tab:blue','tab:green','tab:orange'],
                         ['softmax 1e4', 'softmax 1e5', 'softmax 1e6'])

#%%

cifar_weighted_1 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-76samhjh-selected_index_counts_table:v174',
    artifact_name)

cifar_weighted_2 = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-z9y1qj5j-selected_index_counts_table:v174',
    artifact_name)

plot_diversity_selection([cifar_weighted_1, cifar_weighted_2],
                         [3,3],
                         [1, 0.7],
                         ['tab:blue','tab:green'],
                         ['1', '2'])

#%%


#%%
plot_diversity_selection([ rank_1e4, rank_100, rank_1, cifar_imbal],
                         [5,5,5,5],
                         [1, 0.7, 0.7, 0.7],
                         ['tab:red', 'tab:green', 'tab:orange', 'tab:blue'],
                         ['imbalanced', 'beta = 1e4', 'beta = 100', 'beta=1'])

#%%
plot_diversity_selection([rank_1e4, rank_1e4, rank_1e4],
                         [3,1,9],
                         [1, 0.7, 0.7],
                         ['tab:blue','tab:green', 'tab:orange'],
                         ['3', '1', '9'])

#%% Load CIFAR2 diversity datasets:
    
cifar2_success = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-q3e9pyd6-selected_index_counts_table:v174',
    'selected_index_counts_table')

cifar2_failed = load_run(
    'william_bankes/Robust RHO Parallel Runs/run-qltrnonk-selected_index_counts_table:v174',
    'selected_index_counts_table')

#%%

plot_diversity_selection([cifar2_success, cifar2_failed],
                         [0,0],
                         [1,0.7],
                         ['tab:green', 'tab:red'],
                         ['success', 'failed'])

