# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:21:09 2023

@author: William
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt


def load_run_history(entity_name):
    api = wandb.Api()
    run = api.run(entity_name)
    return run.history(samples=1e6) #set samples as high as possible


def process_multiple_run_history(run_histories, field):
  
    processed_runs = []  
  
    for run in run_histories:
        
        #Filter the NaN entries from the model run:
        df = run[field]
        
        _filter = df.isnull()
        df_filtered = df[~_filter]
        
        processed_runs.append(df_filtered)
        
    #Stack together processed runs:
    df = pd.concat(processed_runs, axis=1)
    
    #Calculate the mean and std
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    
    return mean, std
    
def plot_multiple_run_history( run_mean_std, title, labels, colors, xlabel='steps', ylabel='loss'):
    
    fig, axs = plt.subplots()
    
    for i, (mean, std) in enumerate(run_mean_std):
        
        #Plot the mean and std:     
        axs.plot(mean.index, mean, label=labels[i], color=colors[i])
        axs.fill_between(mean.index, mean-std, mean+std, alpha=0.2, color=colors[i])
    
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    axs.set_title(title)  
    plt.legend()
    
    return fig, axs
    
#%% Can in theory query by filtering the correct tags/properties of the run
        
api = wandb.Api()
runs = api.runs(
 path="william_bankes/Robust RHO Parallel Runs",
 filters=[{"tags": "CIFAR2"}, {"tags": "RHO-Loss"}],
)



#%% Load CIFAR2 runs:
    
entity_names = [
    'william_bankes/Robust RHO Parallel Runs/drx3hwnq',
    'william_bankes/Robust RHO Parallel Runs/xgfj4vhs',
    'william_bankes/Robust RHO Parallel Runs/aztclk9z',
    'william_bankes/Robust RHO Parallel Runs/9fozlcx1',
    'william_bankes/Robust RHO Parallel Runs/jc2xoacn',
    'william_bankes/Robust RHO Parallel Runs/dngfk6d6',
    'william_bankes/Robust RHO Parallel Runs/xgfj4vhs',
    'william_bankes/Robust RHO Parallel Runs/aztclk9z',
    'william_bankes/Robust RHO Parallel Runs/qltrnonk',
    ]
cifar2_fails = set(list(entity_names))
cifar2_successes = [
    'william_bankes/Robust RHO Parallel Runs/q3e9pyd6',
    'william_bankes/Robust RHO Parallel Runs/rrrqzg3v']

cifar2_uniform = [
    'william_bankes/Robust RHO Parallel Runs/05ybysuc',
    'william_bankes/Robust RHO Parallel Runs/bbwngxe4',
    'william_bankes/Robust RHO Parallel Runs/3vfjfwkv',
    'william_bankes/Robust RHO Parallel Runs/5o5lp0f7',
    'william_bankes/Robust RHO Parallel Runs/0ggkrr7z']

#%% CIFAR 2 train loss epoch

fail_runs = [load_run_history(name) for name in entity_names]
fail_mean_std = process_multiple_run_history(fail_runs, 'train_loss_epoch')

success_runs = [load_run_history(name) for name in cifar2_successes]
success_mean_std = process_multiple_run_history(success_runs, 'train_loss_epoch')

uniform_runs = [load_run_history(name) for name in cifar2_uniform]
uniform_mean_std = process_multiple_run_history(uniform_runs, 'train_loss_epoch')

plot_multiple_run_history([success_mean_std, fail_mean_std, uniform_mean_std], 
                          'train_loss_epoch',
                          labels=['success', 'fail', 'uniform'],
                          colors=['tab:green', 'tab:red', 'tab:orange'])

#%% Train Acc 

fail_runs = [load_run_history(name) for name in entity_names]
fail_mean_std = process_multiple_run_history(fail_runs, 'train_acc_epoch')

success_runs = [load_run_history(name) for name in cifar2_successes]
success_mean_std = process_multiple_run_history(success_runs, 'train_acc_epoch')

uniform_runs = [load_run_history(name) for name in cifar2_uniform]
uniform_mean_std = process_multiple_run_history(uniform_runs, 'train_acc_epoch')

plot_multiple_run_history([success_mean_std, fail_mean_std, uniform_mean_std],
                          'train_acc_epoch',
                          labels=['success', 'fail', 'uniform'],
                          colors=['tab:green', 'tab:red', 'tab:orange'], ylabel='accuracy')


#%% Validation Loss 

fail_runs = [load_run_history(name) for name in entity_names]
fail_mean_std = process_multiple_run_history(fail_runs, 'val_loss_epoch')

success_runs = [load_run_history(name) for name in cifar2_successes]
success_mean_std = process_multiple_run_history(success_runs, 'val_loss_epoch')

uniform_runs = [load_run_history(name) for name in cifar2_uniform]
uniform_mean_std = process_multiple_run_history(uniform_runs, 'val_loss_epoch')

plot_multiple_run_history([success_mean_std, fail_mean_std, uniform_mean_std],
                          'val_loss_epoch',
                          labels=['success', 'fail', 'uniform'],
                          colors=['tab:green', 'tab:red', 'tab:orange'],
                          ylabel='accuracy')

#%% Val Acc

fail_runs = [load_run_history(name) for name in entity_names]
fail_mean_std = process_multiple_run_history(fail_runs, 'val_acc_epoch')

success_runs = [load_run_history(name) for name in cifar2_successes]
success_mean_std = process_multiple_run_history(success_runs, 'val_acc_epoch')

uniform_runs = [load_run_history(name) for name in cifar2_uniform]
uniform_mean_std = process_multiple_run_history(uniform_runs, 'val_acc_epoch')

plot_multiple_run_history([success_mean_std, fail_mean_std, uniform_mean_std],
                          'val_acc_epoch',
                          labels=['success', 'fail', 'uniform'],
                          colors=['tab:green', 'tab:red', 'tab:orange'], 
                          ylabel='accuracy')




