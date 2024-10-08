# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 09:39:07 2023

@author: William
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt 


def download_datasets(dir_path):
    
    file_paths = []
    
    #find files in dir:
    for file in os.listdir(dir_path):
        if file.endswith(".pt"):
            file_paths.append(os.path.join(dir_path, file))
    
    return [torch.load(f) for f in file_paths], file_paths


def process_file_path_names(file_paths):
    return [file.split('\\')[-1] for file in file_paths]
    

def process_loaded_output(loaded_output):
    
    irred_losses = loaded_output['irreducible_losses']
    sorted_targets = loaded_output['sorted_targets']
    output_filter = irred_losses.isnan()
    
    data = torch.concat([irred_losses[~output_filter][None,:],
                         sorted_targets[~output_filter][None,:]], axis=0).T
    df_data = pd.DataFrame(data.numpy(), columns=['losses', 'targets'])   
    
    grp_data = df_data.groupby('targets').agg({'losses':'mean'})
    
    return grp_data


if __name__ == '__main__':
    
    file_path = r'C:\Users\William\Documents\Programming\PhD\Datasets\class_robust_irred_models'
    
    datasets, file_paths = download_datasets(file_path)
    file_names = process_file_path_names(file_paths)
    grp_data = [process_loaded_output(data) for data in datasets]
    
    #Find the standard losses:
    std_data_index = file_names.index('irred_losses_and_checks_100.pt')
    std_name = file_names[std_data_index]
    std_data = grp_data[std_data_index]

    #remove from visualised data:    
    del grp_data[std_data_index], file_names[std_data_index]
       
    
    #Plotting
    for i, data in enumerate(grp_data):
        
        plt.bar(data.index, data['losses'],
                width=0.4, label='c = ' + file_names[i][-4])
        plt.bar(std_data.index + 0.4, std_data['losses'],
                width=0.4, label=std_name)
        plt.legend()
        plt.title('Irreducible Model Loss per Class for $\mathcal{D}_{ho}^{(c)}$')
        plt.xlabel('Class Label')
        plt.ylabel('Mean Model Loss per class')
        plt.ylim([0,1.2])
        plt.show()
        

