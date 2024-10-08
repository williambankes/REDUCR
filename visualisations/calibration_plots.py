# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:31:25 2023

@author: William
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

    
def download_datasets(dir_path):
    
    file_paths = []
    
    #find files in dir:
    for file in os.listdir(dir_path):
        if file.endswith(".pt"):
            file_paths.append(os.path.join(dir_path, file))
    
    
    return [torch.load(f) for f in file_paths]


irred_dataset = download_datasets(
    r'C:\Users\William\Documents\Programming\PhD\Datasets\Robust_RHO_Project\model_calibration\irred_model')

target_dataset = download_datasets(
    r'C:\Users\William\Documents\Programming\PhD\Datasets\Robust_RHO_Project\model_calibration\target_model')

#%% Plot model calibration for CIFAR2:

def prep_data(data, temp=1.):

    temperature = torch.tensor(temp, dtype=torch.float)    

    #Remove all zero rows:
    zero_filter = (data['logits'].sum(axis=-1) != 0)
    logits = data['logits'][zero_filter]
    targets = data['sorted_targets'][zero_filter]
    
    #Select the positive class:
    return targets, torch.softmax(logits/temperature, dim=-1)[:,-1]
    
prep_data = [prep_data(data, temp=1.) for data in irred_dataset]
calib_data = [calibration_curve(data[0], data[1]) for data in prep_data]


fig, axs = plt.subplots()
#for data in calib_data:    
#axs.plot(data[0], data[1])

i = 1

axs.plot(calib_data[i][0], calib_data[i][1])    
    
    
axs.set_xlabel('Portion of Samples whose predictions are correct')
axs.set_ylabel('Mean Predicted Probability of Each Bin')
axs.set_title('Calibration Plot of CIFAR2 Target/Irreducible Model Results')

axs.plot(np.linspace(0,1,20), np.linspace(0,1,20),
         'r', lw=2, label='Ideal Calibration')

plt.legend()
