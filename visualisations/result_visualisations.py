# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:37:58 2022

- Add some dataloader that loads all files and creates 'df'
- Loads 'df' and creates error bar plots -> what function to use?
    - yerr argument in the plt.bar(**, yerr, **)
    - put in the std of the losses across the 'Runs/Batch' axis


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
    
    
    return [torch.load(f) for f in file_paths]

def process_loaded_output(loaded_output):
    
    #data = torch.concat([(loaded_output['sorted_targets'] == loaded_output['preds'])[None,:],
    #                     loaded_output['sorted_targets'][None,:]], axis=0).T
    
    data = torch.concat([loaded_output['irreducible_losses'][None,:],
                         loaded_output['sorted_targets'][None,:]], axis=0).T
    df_data = pd.DataFrame(data.numpy(), columns=['losses', 'targets'])   
    
    grp_data = df_data.groupby('targets').agg({'losses':'mean'})
    
    return grp_data

def process_grouped_output(processed_data):
    
    grp_output = pd.concat(processed_data)
    
    grp_agg = grp_output.reset_index().\
        groupby('targets').agg({'losses':['mean', 'std']})
        
    return grp_agg
    
    
def plot_losses(data, title, labels=None, width=None, yerr=None, colours=None):
    
    fig, axs = plt.subplots(figsize=(10,7))
    width=0.2
    
    if isinstance(data, list):
        for i, df in enumerate(data):
            axs.bar(df.index + (width*i), 
                    df['losses']['mean'],
                    yerr= df['losses']['std'] if yerr is not None else yerr,
                    width=width,
                    label=labels[i],
                    color=colours[i] if colours is not None else None)
            
        axs.set_xticks(data[0].index+width*(len(data)//2),
                       data[0].index)
        axs.set_ylim([0,1.1])
        axs.legend()

        
    else:    
        axs.bar(data.index, data['losses']['mean'], 
                yerr=data['losses']['std'])
        
    axs.set_xlabel('Class label')
    axs.set_ylabel('Mean Losses Across Multiple Runs')
    axs.set_title(title)
    
def download_and_plot_dataset(dir_paths, title, 
                              labels=None, width=None,
                              yerr=None, colours=None):
    
    datasets = []
    for dir_path in dir_paths:
        dataset = download_datasets(dir_path)
        processed_data = [process_loaded_output(data) for data in dataset]
        datasets.append(process_grouped_output(processed_data))
            
    plot_losses(datasets, title, labels=labels, 
                width=width, yerr=yerr, colours=colours)
    

if __name__ == '__main__':
       

    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\uniform_selection_results')
    processed_data = [process_loaded_output(data) for data in datasets]
    grp_data_uniform = process_grouped_output(processed_data)
    
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\reducible_loss_selection_results')
    processed_data = [process_loaded_output(data) for data in datasets]
    grp_data_rho = process_grouped_output(processed_data)
    
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\reducible_loss_selection_results_im')
    processed_data = [process_loaded_output(data) for data in datasets]
    grp_data_rho_im = process_grouped_output(processed_data)
    
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\uniform_selection_results_im')
    processed_data = [process_loaded_output(data) for data in datasets]
    grp_data_uniform_im = process_grouped_output(processed_data)
        
    #Can be used to plot multiple datasets on the same graph:
    plot_losses([grp_data_uniform,
                 grp_data_uniform_im,
                 grp_data_rho,
                 grp_data_rho_im],
                'Comparative performance of different selection methods on different datasets',
                ['Uniform Selection: Balanced',
                 'Uniform Selection: Imbalanced',
                 'RHO Selection: Balanced','RHO Selection: Imbalanced'],
                yerr=True,
                colours=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    
    #Download single data and process:       
    datasets = download_datasets( #Add new data here to compare with the uniform and rho methods..
        r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_08')
    processed_data = [process_loaded_output(data) for data in datasets]
    grp_data_robust_eta1 = process_grouped_output(processed_data)
        
    plot_losses([grp_data_uniform,
                 grp_data_rho,
                 grp_data_robust_eta1],
                'Comparative performance of balanced and unbalanced datasets',
                ['Uniform Selection (10 runs)',
                 'RHO Selection (10 runs)',
                 'Weighted 3 irred 80% Class 3(2 Runs)'],
                #'Fixed weights 3,4 & 6'],
                yerr=True,
                colours=['tab:purple', 'tab:orange', 'tab:green'])
#%%

    #Plot 2 class setting:
    data = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\robust_selection_tests\cifar2_uni')
    processed_data = [process_loaded_output(data[0])]
    cifar2_uni = process_grouped_output(processed_data)
    
    data = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\robust_selection_tests\cifar2_rho')
    processed_data = [process_loaded_output(data[0])]
    cifar2_rho = process_grouped_output(processed_data)
    
    
    plot_losses([cifar2_uni,
                cifar2_rho,
                ],
                'Comparative performance of uniform and rho selection methods on CIFAR2',
                ['uniform',
                 'rho',
                 ],
                width=0.05)
    
#%%
    #Plot general comparison of hyperparam performance
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_02')
    processed_data = [process_loaded_output(data) for data in datasets]
    cifar10_weight_02 = process_grouped_output(processed_data)
    
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_04')
    processed_data = [process_loaded_output(data) for data in datasets]
    cifar10_weight_04 = process_grouped_output(processed_data)
    
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_06')
    processed_data = [process_loaded_output(data) for data in datasets]
    cifar10_weight_06 = process_grouped_output(processed_data)
    
    datasets = download_datasets(
        r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_08')
    processed_data = [process_loaded_output(data) for data in datasets]
    cifar10_weight_08 = process_grouped_output(processed_data)
    
    datasets = download_datasets( #Add new data here to compare with the uniform and rho methods..
        r'C:\Users\William\Documents\Programming\PhD\Datasets\robust_selection_tests\weighted_3')
    processed_data = [process_loaded_output(data) for data in datasets]
    weighted_3 = process_grouped_output(processed_data)
    
    datasets = download_datasets( #Add new data here to compare with the uniform and rho methods..
        r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\irred_3_results')
    processed_data = [process_loaded_output(data) for data in datasets]
    irred_3 = process_grouped_output(processed_data)
    
    plot_losses([#cifar10_weight_02,
                cifar10_weight_04,
                cifar10_weight_06,
                cifar10_weight_08,
                ],
                '',
                [#'irred class represented 20%',
                 'weighted 3 irred class represented 40%',
                 'weighted 3 irred class represented 60%',
                 'weighted 3 irred class represented 80%',

                 ], yerr=True, width=2,
                colours=['tab:blue', 'tab:orange', 'tab:green'])
    
    plot_losses([irred_3,
                 grp_data_rho,
                 cifar10_weight_08],
                '',
                [
                 'class 3 irred model, irred class percent 46% (5 runs) ',
                 'RHO-Loss (10 runs)',
                 'weighted 3 irred class percent 80% (5 runs)'
                 ], yerr=True, width=2, colours=['tab:purple', 'tab:orange', 'tab:green'])

#%% Plot the class losses for the diversity results:

    datasets = download_datasets( #Add new data here to compare with the uniform and rho methods..
        r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results\cifar_weighted_random')
    processed_data = [process_loaded_output(data) for data in datasets]
    cifar_weighted_sampling = process_grouped_output(processed_data)
    
    datasets = download_datasets( #Add new data here to compare with the uniform and rho methods..
        r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results\cifar_uniform_select_im2')
    processed_data = [process_loaded_output(data) for data in datasets]
    cifar_imbal = process_grouped_output(processed_data)
    
        
    plot_losses([
                 cifar_weighted_sampling,
                 grp_data_robust_eta1,
                 cifar_imbal,],
                'Comparison of Model performance on Class 3',
                ['Uniform Selection (10 Runs)',
                 'Weighted RHO-Loss, Class 3 power sampling (2 Runs)',
                 'Weighted RHO-Loss, Class 3 no sampling (5 runs)',
                 'Imbalanced Sampling Class 3 (2 Runs)'],
                yerr=True,
                colours=['tab:orange', 'tab:blue', 'tab:green'])

#%% Plot diversity class performance for different sampling methods

diversity_root = r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results'
diversity_dir_names = ['irred_results',
                       'cifar_weighted_softmax_1',
                       'cifar_weighted_softmax_100',
                       'cifar_weighted_softmax_1e6']
diversity_dir_paths = [os.path.join(diversity_root, dir_name) for dir_name in diversity_dir_names]
#diversity_dir_paths.append(
#    r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_08')

download_and_plot_dataset(diversity_dir_paths, 
                          'CIFAR10 Weighted RHO-Loss, Class 3 Softmax Sampling method',
                          labels=['irreducible_model results',
                                  'beta=1',
                                  'beta=100',
                                  'beta=1e6'],
                          yerr=True,
                          width=2)
    
#%%

diversity_root = r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results'
diversity_dir_names = ['irred_results',
                       'cifar_weighted_rank_1',
                       #'cifar_weighted_rank_10',
                       'cifar_weighted_rank_100',
                       'cifar_weighted_rank_1e4']
diversity_dir_paths = [os.path.join(diversity_root, dir_name) for dir_name in diversity_dir_names]
#diversity_dir_paths.append(
#    r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results\weighted_3_irred_08')

download_and_plot_dataset(diversity_dir_paths, 
                          'CIFAR10 Weighted RHO-Loss, Class 3 Soft-Rank Sampling method',
                          labels=['irreducible_model results',
                                  'beta=1',
                                  #'beta=10',
                                  'beta=100',
                                  'beta=1e4'],
                          yerr=True,
                          width=2)

#%%


diversity_root = r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results'
diversity_dir_names = ['irred_3_results',
                       'irred_3_90_results',
                       'weighted_3_irred_08']                    
diversity_dir_paths = [os.path.join(diversity_root, dir_name) for dir_name in diversity_dir_names]

download_and_plot_dataset(diversity_dir_paths, 
                          'CIFAR10 Weighted RHO-Loss, Class 3 Softmax Sampling method',
                          labels=['irred results, 46%',
                                  'irred results, 90%',
                                  'weighted rho, 80% irred'],
                          yerr=True,
                          width=2)

#%%

diversity_root = r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results'
diversity_dir_names = ['cifar_uniform_select_im2',
                       'cifar_weighted_top_k',
                       'cifar_weighted_rank_1e4',
                       'cifar_weighted_softmax_100']                    
diversity_dir_paths = [os.path.join(diversity_root, dir_name) for dir_name in diversity_dir_names]

download_and_plot_dataset(diversity_dir_paths, 
                          'CIFAR10 Models learnt to improve performance of class 3',
                          labels=['imbalanced',
                                  'weighted rho top k',
                                  r'weighted rho rank, $\beta=1e4$',
                                  r'weighted rho softmax $\beta=100$'],
                          yerr=True,
                          width=2)

#%% Opening Graph

diversity_root = r'C:\Users\William\Documents\Programming\PhD\Datasets\diversity_results'
diversity_dir_names = ['cifar_uniform_select_im2',
                       'cifar_weighted_top_k',
                       'cifar_weighted_top_k_pt_02']                    
diversity_dir_paths = [os.path.join(diversity_root, dir_name) for dir_name in diversity_dir_names]

root2 = r'C:\Users\William\Documents\Programming\PhD\Datasets\weighted_rho_results'
diversity_dir_paths.append(os.path.join(root2, 'weighted_3_irred_08'))


download_and_plot_dataset(diversity_dir_paths, 
                          'CIFAR10 Comparison of Imbalanced and Weighted RHO on class 3',
                          labels=['imbalanced',
                                  'weighted rho, class 3, irred 46% imbalanced',
                                  'weighted rho, class 3 percent train 20%',
                                  'weighted rho, class 3, irred 80% imbalanced'],
                          yerr=True,
                          width=2)

#%%
cifar2_root = r"C:\Users\William\Documents\Programming\PhD\Datasets\Robust_RHO_Project\CIFAR2"
cifar2_names = ['rho', 'uniform', 'rho_success', 'rho_fail']
paths = [os.path.join(cifar2_root, name) for name in cifar2_names]

download_and_plot_dataset(paths,
                          'CIFAR2 Comparison of RHO-Loss selection and Uniform Loss Selection',
                          labels=['rho', 'uniform', 'rho_success', 'rho_fail'],
                          yerr=True,
                          width=0.5)

    
    
    