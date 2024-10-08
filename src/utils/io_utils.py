# -*- coding: utf-8 -*-
import os

import torch
import gcsfs

import pytorch_lightning as pl

from typing import Union
from abc import ABC, abstractmethod

from src.models.OneModel import OneModel
from src.models.MultiModels import MultiModels
from src.utils.utils import get_logger

log = get_logger(__name__)

class IOFactory:
          
    def __init__(self, env:str,
                        datamodule_name:str, #get from datamodule/or define manually
                        number:int, 
                        home_dir: Union[None, str] = None,
                        selection_method: Union[None, str] = None, #get from selection method
                        imbalanced:bool = False, #Figure out how to manage this variable...
                        imbalanced_class: Union[None,int] = None, #get from sampler factory 
                        gradient_weighted_class: Union[None, int] = None, #get from model
                        multi_headed_model: bool = False, #get from model
                        gcp_project: Union[None, str] = None, 
                        gcp_storage:Union[None, str] = None,
                        input_dir: Union[None, str] = None, #input dir to read from in cluster io
                        output_dir: Union[None, str] = None, #output dir to save to in cluster io
                        eta: Union[None, float] = None, #get from selection method
                        beta: Union[None, float] = None, #get from selection method
                        ):
        
        #Store variables locally:
        self.home_dir = home_dir
        self.selection_method = selection_method
        self.datamodule_name = datamodule_name
        self.number = number
        self.imbalanced = imbalanced
        
        print('gradient_weighted_class', gradient_weighted_class)
        print('class_imbalance', imbalanced_class)
        print('multi_headed_model', multi_headed_model)
        
        #XOR of the three options when at least one of them is not None
        if (gradient_weighted_class is not None) or (imbalanced_class is not None) or (multi_headed_model):
            assert ((gradient_weighted_class is not None) != (imbalanced_class is not None)) != multi_headed_model,\
                 f"""Only one of gradient_weighted_class:{gradient_weighted_class}, 
                 multi_headed_model:{multi_headed_model}, and imbalanced_class: {imbalanced_class} can be not None\False."""

        if imbalanced_class is not None:
            class_imbalance = imbalanced_class
        elif gradient_weighted_class is not None:
            class_imbalance = gradient_weighted_class
        elif multi_headed_model:
            class_imbalance = 'mh'
            print('IO using multi_headed_model')
        else:
            class_imbalance = None
        
        self.class_imbalance = class_imbalance
                      
        #Env specific variables
        self.gcp_project = gcp_project
        self.gcp_storage = gcp_storage
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.env = env
        
        #Selection method relevant hyperparams:
        if selection_method == 'robust_reducible_loss_selection': 
            assert eta is not None, 'eta must not be None for robust reducible loss selection'
         
        #Hyperparameters
        self.eta = eta
        self.beta = beta        
        
    def create_model_io(self):
        
        if self.env == 'local':
            return LocalIO(
                home_dir=self.home_dir,
                selection_method=self.selection_method,
                datamodule_name=self.datamodule_name,
                number=self.number,
                imbalanced=self.imbalanced,
                class_imbalance=self.class_imbalance,
                eta=self.eta,
                beta=self.beta)
        
        elif self.env == 'gcp':
            
            #assert values are provided for default arguments:
            assert (self.gcp_project is not None) and (self.gcp_storage is not None),\
                "Values must be provided for gcp_project and gcp_storage in io_model config"
            
            if self.home_dir is not None:
                log.warning("IOFactory: env='gcp' and home_dir is not None")
            
            
            return GoogleCloudPlatformIO(
                selection_method=self.selection_method,
                datamodule_name=self.datamodule_name,
                number=self.number,
                imbalanced=self.imbalanced,
                class_imbalance=self.class_imbalance,
                gcp_project=self.gcp_project,
                gcp_storage=self.gcp_storage,
                eta=self.eta,
                beta=self.beta)
        
        elif self.env == 'cluster':
            
            #assert value provided:
            assert (self.output_dir is not None) and (self.input_dir is not None),\
                "Values must be provided for output_dir and input_dir in io_model config"
                
            if self.home_dir is not None:
                log.warning("IOFactory: env='cluster' and home_dir is not None")
                
            return ClusterIO(output_dir=self.output_dir,
                            input_dir=self.input_dir,
                            selection_method=self.selection_method,
                            datamodule_name=self.datamodule_name,
                            number=self.number, 
                            imbalanced=self.imbalanced,
                            class_imbalance=self.class_imbalance,
                            eta=self.eta,
                            beta=self.beta)
            
        else:
            #If the environment is not accounted for raise an error:
            raise NotImplementedError(f"env: {self.env} is not accounted for by IOFactory")

            

class IO(ABC):

    def __init__(self, imbalanced: bool,
                 class_imbalance: Union[None, int],
                 number: int, 
                 home_dir: str,
                 datamodule_name: str,
                 selection_method: str,
                 eta: Union[None, float],
                 beta: Union[None, float]):
        """
        IO standardises the file naming convention across different environments whilst
        abstracting the specifics of how the loading and saving occur in each case

        Parameters
        ----------
        imbalanced : bool
            Is the dataset imbalanced or not
        class_imbalance : int | None
            Which class is the dataset imbalanced towards, None if no class imbalanced 
        number : int
            Unique identifying number in multiple model runs
        home_dir: str
            Path to home directory where file load/save structure created. In the case
            of GCP this should be the storage bucket gs://<bucket name here>
        datamodule_name: str
            Name of the dataset under which to load/save the relevant experiment data
        selection_method: str
            Name of the method of selection used in the experiment
        <hyperparameters>: Union[None, <type>]
            Hyperparameters for different experiments that appear in the naming of the target
            model            

        Returns
        -------
        None.

        """
        
        #File name variables:
        self.imbalanced = imbalanced
        self.number = number
        self.class_imbalance = class_imbalance
        self.selection_method = selection_method
        
        #Selection method specific hyperparams:
        self.eta = eta
        self.beta = beta
        
        #File path variables:
        self.home_dir = home_dir
        self.datamodule_name = datamodule_name


    def create_losses_and_checks_file_name(self,
                                           irreducible_model:bool,
                                           class_imbalance:Union[None, int]) -> str:
        
        imbalanced = '_im' if self.imbalanced else ''
        
        #if class imbalance defined in the inputs then we raise an error
        if self.class_imbalance is not None:
            assert class_imbalance is None, 'class_imbalance defined in IO config file must be set to null'

            #set class imbalance to self.class_imbalance
            class_imbalance = self.class_imbalance
            
        if irreducible_model:
            
            class_imbalance_str = f'_class_{class_imbalance}' \
                if class_imbalance is not None else ''
            
            file_name = "irred_losses_and_checks{0}_{1}{2}.pt".format(imbalanced,
                                                                      self.number,
                                                                      class_imbalance_str)
            
        else:
            
            #Address selection method specific components
            eta_file_name = '_eta_{}'.format(self.eta)\
                if self.selection_method == 'robust_reducible_loss_selection' else ''
            beta_file_name = '_beta_{}'.format(self.beta)\
                if self.beta is not None else ''
                
            #Create file name:
            file_name = "{0}_losses_and_checks{1}_{2}{3}{4}.pt".format(self.selection_method,
                                                                  imbalanced,
                                                                  self.number,
                                                                  eta_file_name,
                                                                  beta_file_name)
            
        print('losses and checks file name', file_name)
            
        return file_name
    
    def create_checkpoint_file_name(self, 
                                    irreducible_model:bool,
                                    class_imbalance: Union[None, int]) -> str:
        
        #if class imbalance defined in the inputs then we raise an error
        if self.class_imbalance is not None:
            assert class_imbalance is None,\
                f"""IO object instantiated with non-None class_imbalance: {class_imbalance},
                self.class_imbalance:{self.class_imbalance} set to None"""
            #set class imbalance to self.class_imbalance
            class_imbalance = self.class_imbalance
        
        if irreducible_model:
            
            class_imbalance_str = f'_class_{class_imbalance}'\
                if class_imbalance is not None else ''
            checkpoint_file_name = "irred_model_checkpoint_{0}{1}.ckpt".format(self.number,
                                                                               class_imbalance_str)
        else:
            
            #Address selection method specific components
            eta_file_name = '_eta_{}'.format(self.eta)\
                if self.selection_method == 'robust_reducible_loss_selection' else ''
            beta_file_name = '_beta_{}'.format(self.beta)\
                if self.beta is not None else ''
            
            #Create the file name:
            checkpoint_file_name = "{0}_model_checkpoint_{1}{2}{3}.pt".\
                format(self.selection_method,
                self.number,
                eta_file_name,
                beta_file_name)
                        
        return checkpoint_file_name
    
    def create_file_path(self, file_name:str, irreducible_model:bool) -> str:
        
        if irreducible_model:
            model = 'irreducible_models'
        else:
            model = 'target_models'
        
        file_path = os.path.join(self.home_dir, self.datamodule_name, 
                                 model, file_name)
        
        return file_path
    
    def create_load_checkpoint_path(self, irreducible_model:bool) -> str:
        
        checkpoint_file_name = self.\
            create_checkpoint_file_name(irreducible_model=irreducible_model,
                                        class_imbalance=None)
        checkpoint_file_path = self.create_file_path(checkpoint_file_name,
                                                     irreducible_model)
        
        return checkpoint_file_path
    
    @abstractmethod
    def load_checkpoint(self, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None) -> pl.LightningModule:
        pass
    
    @abstractmethod
    def load_losses_and_checks(self, 
                               irreducible_model:bool,
                               class_imbalance:Union[None, int]=None) -> torch.Tensor:
        pass
        
    @abstractmethod
    def save_checkpoint(self, checkpoint, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None) -> None:
        pass
        
    @abstractmethod
    def save_losses_and_checks(self,
                               losses_and_checks:dict, irreducible_model:bool,
                               class_imbalance:Union[None, int]=None) -> None:
        pass        

class LocalIO(IO):
    
    def __init__(self, 
                 home_dir:str,
                 selection_method:str,
                 datamodule_name:str,
                 number:int, 
                 imbalanced:bool,
                 class_imbalance: Union[None,int],
                 eta: Union[None, float],
                 beta: Union[None, float]) -> None:
        
                
        super().__init__(imbalanced=imbalanced, 
                         class_imbalance=class_imbalance, 
                         number=number,
                         home_dir=home_dir,
                         datamodule_name=datamodule_name,
                         selection_method=selection_method,
                         eta=eta,
                         beta=beta)
                
             
    def save_losses_and_checks(self, losses_and_checks, irreducible_model:bool,
                               class_imbalance:Union[None, int]=None) -> None:
        
        #Create the file path:
        losses_and_checks_file_name =\
            self.create_losses_and_checks_file_name(irreducible_model=irreducible_model,
                                                    class_imbalance=class_imbalance)
        losses_and_checks_file_path = self.create_file_path(losses_and_checks_file_name,
                                                            irreducible_model)
        
        #Save the losses and checks to the relevant file path:
        torch.save(losses_and_checks, losses_and_checks_file_path)
        
    def save_checkpoint(self, checkpoint, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None) -> None:
                
        #Create the file path
        checkpoint_file_name = self.create_checkpoint_file_name(irreducible_model=irreducible_model,
                                                                class_imbalance=class_imbalance)
        checkpoint_file_path = self.create_file_path(checkpoint_file_name,
                                                     irreducible_model)
        
        #Save the model to the relevant file path
        torch.save(checkpoint, checkpoint_file_path)
        
    def load_losses_and_checks(self, irreducible_model: bool,
                               class_imbalance:Union[None, int]=None) -> dict:
        
        #Create the file path:
        losses_and_checks_file_name = self.\
            create_losses_and_checks_file_name(irreducible_model=irreducible_model,
                                                class_imbalance=class_imbalance)
        losses_and_checks_file_path = self.create_file_path(losses_and_checks_file_name,
                                                            irreducible_model)
        
        #Save the losses and checks to the relevant file path:
        return torch.load(losses_and_checks_file_path)
    
    def load_checkpoint(self, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None):
        
        #Create the file path:
        checkpoint_file_name = self.\
            create_checkpoint_file_name(irreducible_model=irreducible_model,
                                        class_imbalance=class_imbalance)
        checkpoint_file_path = self.create_file_path(checkpoint_file_name,
                                                     irreducible_model)
        
        print(f'Loading checkpoint from {checkpoint_file_path}')
        
        #Load the model checkpoints:
        if irreducible_model:
            return OneModel.load_from_checkpoint(checkpoint_file_path) 
        else:
            return MultiModels.load_from_checkpoint(checkpoint_file_path)
        
class GoogleCloudPlatformIO(IO):
        
    def __init__(self,
                 selection_method:str,
                 datamodule_name:str,
                 number:int, 
                 imbalanced:bool,
                 class_imbalance: Union[None,int],
                 gcp_project: str,
                 gcp_storage: str,
                 eta: Union[None, float],
                 beta: Union[None, float]) -> None:
                    
        #We replace the home_directory with the gcp_storage path
        
        super().__init__(imbalanced=imbalanced, 
                         class_imbalance=class_imbalance, 
                         number=number,
                         home_dir=gcp_storage, #replace the home directory with the gcp storage
                         datamodule_name=datamodule_name,
                         selection_method=selection_method,
                         eta=eta,
                         beta=beta)
                   
        self.gcp_project = gcp_project
        self.gcp_storage = gcp_storage

    def load_checkpoint(self, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None):
        
        #As this is platform independent we can probably move this to the parent class:
        
        #Create the file path:
        checkpoint_file_name = self.\
            create_checkpoint_file_name(irreducible_model=irreducible_model,
                                        class_imbalance=class_imbalance)
        checkpoint_file_path = self.create_file_path(checkpoint_file_name,
                                                     irreducible_model)
        
        #Load the model checkpoints:
        if irreducible_model:
            return OneModel.load_from_checkpoint(checkpoint_file_path) 
        else:
            return MultiModels.load_from_checkpoint(checkpoint_file_path)
    
    def load_losses_and_checks(self, irreducible_model:bool,
                               class_imbalance:Union[None, int]=None) -> torch.Tensor:
        
        #create file path:
        losses_and_checks_file_name = self.\
            create_losses_and_checks_file_name(irreducible_model=irreducible_model,
                                               class_imbalance=class_imbalance)
        losses_and_checks_file_path = self.create_file_path(losses_and_checks_file_name,
                                                            irreducible_model)
        
        #Load file from GCP storage: 
        fs = gcsfs.GCSFileSystem(project=self.gcp_project)

        with fs.open(losses_and_checks_file_path, 'rb') as f:
            losses_and_checks = torch.load(f)
    
        return losses_and_checks
        
    
    def save_checkpoint(self, checkpoint, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None) -> None:
        
        #Create the file path
        checkpoint_file_name = self.\
        create_checkpoint_file_name(irreducible_model=irreducible_model,
                                    class_imbalance=class_imbalance)
        checkpoint_file_path = self.create_file_path(checkpoint_file_name,
                                                     irreducible_model)
        
        #Setup file save system:
        fs = gcsfs.GCSFileSystem(project=self.gcp_project)

        with fs.open(checkpoint_file_path, 'wb') as f:
            torch.save(checkpoint, f)
        

    def save_losses_and_checks(self, losses_and_checks:dict, irreducible_model:bool, 
                               class_imbalance:Union[None, int]=None) -> None:
        
        #Create the file path:
        losses_and_checks_file_name = self.\
            create_losses_and_checks_file_name(irreducible_model=irreducible_model,
                                    class_imbalance=class_imbalance)
        losses_and_checks_file_path = self.create_file_path(losses_and_checks_file_name,
                                                            irreducible_model)
        
        #Setup file save system:
        fs = gcsfs.GCSFileSystem(project=self.gcp_project)

        with fs.open(losses_and_checks_file_path, 'wb') as f:
            torch.save(losses_and_checks, f)
  
class ClusterIO(LocalIO):

    def __init__(self,
                 output_dir:str,
                 input_dir:str,
                 selection_method:str,
                 datamodule_name:str,
                 number:int, 
                 imbalanced:bool,
                 class_imbalance: Union[None,int],
                 eta: Union[None, float],
                 beta: Union[None, float]) -> None:
        """
        We setup an IO class to deal with data on the UCL cluster. This class allows
        data to be read from one place and saved to a new place. This can be utilised
        with temporary data space to save intermediate calculations during training to 
        one directory and complete results to another

        Parameters
        ----------
        output_dir : str
            The directory the save functions should utilise 
        input_dir : str
            The directory the load functions should utilise

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        #Init the parent class
        super().__init__(imbalanced=imbalanced, 
                         class_imbalance=class_imbalance, 
                         number=number,
                         home_dir=input_dir, #Note we set this to input_dir
                         datamodule_name=datamodule_name,
                         selection_method=selection_method,
                         eta=eta,
                         beta=beta)
                           
        self.output_dir = output_dir
        
        paths = list()
        #On the cluster specifically we would like the io_utils to create the relevant files:
        paths.append(os.path.join(self.home_dir, self.datamodule_name))
        paths.append(os.path.join(self.output_dir, self.datamodule_name))
            
        paths.append(os.path.join(self.home_dir, self.datamodule_name,
                                        'irreducible_models'))
        paths.append(os.path.join(self.output_dir, self.datamodule_name,
                                         'target_models'))
            
        for path in paths:
            #check to make sure the path does not exist:
            if not os.path.exists(path):
                log.info(f'ClusterIO Creating directory: {path}')
                os.mkdir(path)        
        
            
    def save_losses_and_checks(self, losses_and_checks, irreducible_model:bool,
                               class_imbalance:Union[None, int]=None) -> None:
        
        #Create the file path:
        losses_and_checks_file_name =\
            self.create_losses_and_checks_file_name(irreducible_model=irreducible_model,
                                                    class_imbalance=class_imbalance)
            
        #CHANGED FROM THE PARENT CLASS TO create_file_path_output
        losses_and_checks_file_path = self.create_file_path_output(losses_and_checks_file_name,
                                                                   irreducible_model)
        
        #Save the losses and checks to the relevant file path:
        torch.save(losses_and_checks, losses_and_checks_file_path)
        
    def save_checkpoint(self, checkpoint, irreducible_model:bool,
                        class_imbalance:Union[None, int]=None) -> None:
                
        #Create the file path
        checkpoint_file_name = self.create_checkpoint_file_name(irreducible_model=irreducible_model,
                                                                class_imbalance=class_imbalance)
        
        #CHANGED FROM THE PARENT CLASS TO create_file_path_output
        checkpoint_file_path = self.create_file_path_output(checkpoint_file_name,
                                                            irreducible_model)
        
        #Save the model to the relevant file path
        torch.save(checkpoint, checkpoint_file_path)
        
    def create_file_path_output(self, file_name:str, irreducible_model:bool) -> str:
        """
        We have created a create_file_path_output method to specifically enable
        the creation of a file path with the output_dir used instead of the input_dir
        which is fed to the home_dir
        """
        
        if irreducible_model:
            model = 'irreducible_models'
        else:
            model = 'target_models'
        
        file_path = os.path.join(self.output_dir, self.datamodule_name, 
                                 model, file_name)
        
        return file_path
        
        