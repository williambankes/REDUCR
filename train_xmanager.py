# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:48:49 2023

@author: William
"""

import os
import argparse


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="")
    
    parser.add_argument("-i", "--irreducible",
                        help="If enabled run the irreducible training script",
                        action="store_true" #default=False, type=Bool
                        )
    parser.add_argument("-e", "--experiment",
                        help="experiment config file to run",
                        default=None,
                        type=str)
    parser.add_argument("-ef", "--experiment_folder",
                        help="directory with experiment config files",
                        default='experiment',
                        type=str)
    parser.add_argument("-io", "--model_io",
                        help="model_io config setup",
                        default="gcp_default",
                        type=str)
    parser.add_argument("-g", "--gpus",
                        help="Override the gpu flag within the base experiment script",
                        default=1,
                        type=int)
    parser.add_argument("-n", "--number",
                        help="Number that uniquely defines experiment",
                        default=1,
                        type=int)
    parser.add_argument("-s", "--seed",
                        help="Seed for random no. generators",
                        default=12,
                        type=int)
    parser.add_argument("-c1", '--command_one', default='', type=str)
    parser.add_argument("-c2", '--command_two', default='', type=str)
    parser.add_argument("-c3", '--command_three', default='', type=str)
    parser.add_argument("-c4", '--command_four', default='', type=str)
    parser.add_argument("-c5", '--command_five', default='', type=str)
    parser.add_argument("-c6", '--command_six', default='', type=str)



    args = parser.parse_args()

    #Create specific commands from the cmd arguments:    
    if args.irreducible:
        model_run = '_irreducible'
    else:
        model_run = ''

    if args.experiment is not None:
        experiment_command = f" +{args.experiment_folder}={args.experiment}"
    else:
        experiment_command = ''

    #Override commands:
    override_commands = [f"model_io={args.model_io}",
                        f"++trainer.accelerator={'gpu' if args.gpus>0 else 'cpu'}",
                        f"++trainer.devices={args.gpus}",
                        f"model_io.number={args.number}",
                        f"++seed={args.seed}"]
    override_commands = ' '.join(override_commands)
                            
    additional_commands = [args.command_one, args.command_two, 
                           args.command_three, args.command_four,
                           args.command_five, args.command_six]
    additional_commands = ' '.join(additional_commands)

    print(f'Additional command string {additional_commands}')
        
    #Create the command: 
    command = f"python run{model_run}.py{experiment_command} {override_commands} {additional_commands}"
        
    print(f'Running {command}')
    os.system(command)

    