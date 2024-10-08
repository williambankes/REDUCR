# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:18:35 2023

@author: William
"""

import os 
import torch
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


def process_loaded_output(loaded_output):
        
    data = torch.concat([loaded_output['irreducible_losses'][None,:],
                         loaded_output['sorted_targets'][None,:]], axis=0).T
    df_data = pd.DataFrame(data.numpy(), columns=['losses', 'targets'])   
        
    return df_data


#%%

dataset = download_datasets(
    r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results\irred_results')
df_pro = process_loaded_output(dataset[0])

nan_filter = df_pro.losses.isna()
target_filter3 = df_pro.targets == 3.0
target_filter1 = df_pro.targets == 1.0

bins = np.histogram(np.hstack((df_pro[~nan_filter&target_filter3].losses,
                               df_pro[~nan_filter&target_filter1].losses)), bins=40)[1]

fig, axs = plt.subplots()
axs.hist(df_pro[~nan_filter&target_filter3].losses, bins, label='class 3')
axs.hist(df_pro[~nan_filter&target_filter1].losses, bins, alpha=0.5, label='class 1')
axs.set_xlim([0,7])
axs.set_title('Comparison of the Irreducible Model loss for Class 3 and 1')
plt.legend()

#%%

dataset = download_datasets(
    r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results\cifar_weighted_softmax_100')
df_pro = process_loaded_output(dataset[0])

nan_filter = df_pro.losses.isna()
target_filter3 = df_pro.targets == 3.0
target_filter1 = df_pro.targets == 1.0

bins = np.histogram(np.hstack((df_pro[~nan_filter&target_filter3].losses,
                               df_pro[~nan_filter&target_filter1].losses)), bins=40)[1]

fig, axs = plt.subplots()
axs.hist(df_pro[~nan_filter&target_filter3].losses, bins, label='class 3')
axs.hist(df_pro[~nan_filter&target_filter1].losses, bins, alpha=0.5, label='class 1')
axs.set_xlim([0,7])
axs.set_xlabel('Model loss of point')
axs.set_ylabel('Number of points with Model Loss')
axs.set_title('Comparison of the Target Model loss for Class 3 and 1')
plt.legend()


