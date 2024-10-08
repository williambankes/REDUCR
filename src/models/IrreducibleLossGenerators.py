# -*- coding: utf-8 -*-

import hydra
import numpy as np

import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

from typing import Union
from abc import ABC, abstractmethod
from src.utils.io_utils import IO, LocalIO
from src.utils.utils import verify_correct_dataset_order, get_logger, unmask_config
from src.datamodules.datasets.imbalanced_dataloader import ImbalancedSamplerFactory

log = get_logger(__name__)

#Define a factory class that is specified in the config file and which interprets the arguments:
class IrreducibleLossGeneratorFactory:
    
    def __init__(self, model_io:IO,
                 selection_method:str,
                 update_irreducible:bool = False,
                 num_categories:Union[None, int] = None,
                 weights: Union[None, list] = None,
                 restricted: bool = False,
                 restricted_type: str = 'max',
                 permuted: bool = False,
                 entropy: bool = False,
                 multi_headed_model:bool = False,
                 checkpoint_model:bool = False):
                
        self.model_io = model_io
        self.selection_method = selection_method
        self.update_irreducible = update_irreducible
        self.num_categories = num_categories
        self.weights = weights
        self.restricted = restricted
        self.restricted_type = restricted_type
        self.permuted = permuted
        self.entropy = entropy
        self.multi_headed_model = multi_headed_model
        self.checkpoint_model = checkpoint_model
    
    def create_loss_generator(self):
        """
        Creates the specific IrreducibleLossGenerator for the specific experiment
        setup

        Raises
        ------
        NotImplementedError
            If the selection method has not been implemented and added to the 
            factory we raise a NotImplementedError.

        Returns
        -------
        IrreducibleLossGenerator
            An object that inherits the abstract class IrreducibleLossGenerator 
            with an implementation specific to the experiment

        """
                        
        #Create IrreducibleLossGenerator class:
        if self.update_irreducible: #Doesn't require a loader
            
            if self.selection_method == 'uniform_selection':
                return None
        
            elif self.selection_method == 'reducible_loss_selection':
                return UpdateIrreducibleReducibleLossGenerator(loader=self.model_io)
            
            elif self.selection_method == 'weighted_reducible_loss_selection':
                return UpdateIrreducibleWeightedReducibleLossGenerator(
                    loader=self.model_io,
                    weights=self.weights)
            
            else:
                
                raise NotImplementedError(
                    f"selection method: {self.selection_method} has not been implemented yet for update_irreducible")
        
        elif self.checkpoint_model:
            
            if (self.selection_method == 'uniform_selection') or\
                (self.selection_method == 'ce_loss_selection'):
                return None
            
            elif self.selection_method == 'reducible_loss_selection':
                
                return ModelIrreducibleLossGenerator(loader=self.model_io)
            
            elif self.selection_method == 'weighted_reducible_loss_selection':
                
                raise NotImplementedError('checkpoint model not implemented for weighted_reducible_loss_selection')
                            
            elif (self.selection_method == 'robust_reducible_loss_selection') or \
                (self.selection_method == 'robust_payoff_reducible_loss_selection'):

                if self.multi_headed_model:
                    
                    return MultiHeadedModelIrreducibleLossGenerator(loader=self.model_io,
                                                                    num_categories=self.num_categories)
                    
                    #raise NotImplementedError('multi_headed_model not implemented when checkpoint_model is True')
                
                else:
                    if self.restricted or self.permuted or self.entropy:
                        raise NotImplementedError(f"""checkpoint_model not implemented when one of
                                                  restricted:{self.restricted}, 
                                                  permuted:{self.permuted}
                                                  or entropy:{self.entropy} are True""")
                    
                    return MultiModelIrreducibleLossGenerator(
                                 loader=self.model_io, 
                                 num_categories=self.num_categories)

            else:
                raise NotImplementedError(
                    f"selection method: {self.selection_method} has not been implemented")
        
        else: 
            
            if (self.selection_method == 'uniform_selection') or\
                (self.selection_method == 'ce_loss_selection'):
                return None
            
            elif self.selection_method == 'reducible_loss_selection':
                
                return PrecomputedReducibleLossGenerator(loader=self.model_io)
            
            elif self.selection_method == 'weighted_reducible_loss_selection':
                
                return PrecomputedWeightedReducibleLossGenerator(
                    loader=self.model_io,
                    weights=self.weights,
                    restricted=self.restricted,
                    restricted_type=self.restricted_type,
                    permuted=self.permuted,
                    entropy=self.entropy)
            
            elif (self.selection_method == 'robust_reducible_loss_selection') or \
                (self.selection_method == 'robust_payoff_reducible_loss_selection'):

                if self.multi_headed_model:
                    
                    return PrecomputedRobustMHReducibleLossGenerator(
                        loader=self.model_io, 
                        num_categories=self.num_categories,
                        restricted=self.restricted,
                        restricted_type=self.restricted_type,
                        permuted=self.permuted,
                        entropy=self.entropy)
                
                else:
                
                    return PrecomputedRobustReducibleLossGenerator(
                                 loader=self.model_io, 
                                 num_categories=self.num_categories,
                                 restricted=self.restricted,
                                 restricted_type=self.restricted_type,
                                 permuted=self.permuted,
                                 entropy=self.entropy)

            else:
                raise NotImplementedError(
                    f"selection method: {self.selection_method} has not been implemented")
                           
    
#The Irreducible Loss generator which is abstracted throughout the code base:
class IrreducibleLossGenerator(ABC):
    """
    Abstract class for the irreducible loss generators. This defines the interface
    that the irreducible loss generators should present to the selection_method
    and models
    
    """     
        
    @abstractmethod
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor):
        """
        Calculates a vector of the irreducible losses

        Parameters
        ----------
        global_index : torch.Tensor
            Tensor of the global indices of the points in the inputted data
        data : torch.Tensor
            Tensor of the data 
        target : TYPE
            Tensor of the target labels
        category : TYPE
            Tensor of the groups/categories known at training time

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod
    def assert_device(self, target_device:torch.device):
        """
        Move the irreducible loss generator model(s)/tensor to the target device

        Parameters
        ----------
        target_device : torch.device
            Target device on which the irreducible loss generator should be setup

        Returns
        -------
        None.

        """
        pass
    
class PrecomputedIrreducibleLossGenerator(IrreducibleLossGenerator, ABC):
        
    @abstractmethod
    def check_precomputed_irreducible_losses(self, datamodule_config:dict, test:bool):
        """
        Ensure the precomputed losses are correctly ordered with respect to the 
        global index

        Parameters
        ----------
        datamodule_config : dict
            dataloader config, used to create a temporary dataloader object

        Returns
        -------
        None.

        """
        pass
    
class PrecomputedMultiIrreducibleLossGenerator(PrecomputedIrreducibleLossGenerator, ABC):
    
    
    def __init__(self, loader:LocalIO, restricted:bool,
                 restricted_type:str, permuted:bool, entropy:bool) -> None:
                
        self.loader = loader
        
        self.irreducible_loss_generator = list()
        self.irreducible_losses = list()
        self.logits = list()
                        
        self.restricted = restricted
        self.restricted_type = restricted_type

        #Irred loss model modes        
        self.permuted = permuted
        self.entropy = entropy
                    
        #Ensure only one of the loss model modes is True:
        assert int(self.permuted) + int(self.entropy) + int(self.restricted) <= 1,\
            f'permuted:{self.permuted}; restricted:{self.restricted}; entropy:{self.entropy} only one may be true'
                
        self.non_zero_indices = None
        
        #checking flag
        self.checked = False

    @property        
    def non_zero_indices(self):
        assert self._non_zero_indices is not None, 'non_zero_indices must be set before it is called'        
        return self._non_zero_indices
            
    @non_zero_indices.setter
    def non_zero_indices(self, values):
        #put some assertions here:
        self._non_zero_indices = values
        
    
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        assert self.checked, 'check_precomputed_irreducible_losses must be run first'
                
        irreducible_losses = list()
        
        #Can we stack the self.irreducible_losses vector
        
        # For each irreducible loss generator collate the irreducible losses
        for i, irred_loss_gen in enumerate(self.irreducible_losses):
            
            #Restricted flag:
            if self.restricted:
            
                #Create temporary losses -> could do without as categories don't change
                #thus adjusted irred losses are actually never needed:
                local_irred_loss_gen = irred_loss_gen[global_index]
                    
                temp_losses = torch.zeros(local_irred_loss_gen.shape,
                                          device=local_irred_loss_gen.device)    
                relevant_group_filter = category == self._non_zero_indices[i]

                if self.restricted_type == 'max':
                    restricted_value = local_irred_loss_gen[~relevant_group_filter].max()
                elif self.restricted_type == 'mean':
                    restricted_value = local_irred_loss_gen[~relevant_group_filter].mean()
                else:
                    raise NotImplementedError(
                        f"restricted_type: {self.restricted_type} hasn't been implemented yet")

                temp_losses[relevant_group_filter] = local_irred_loss_gen[relevant_group_filter]
                temp_losses[~relevant_group_filter] = restricted_value
            
                irreducible_losses.append(temp_losses) 
                   
            elif self.permuted:
                
                #Permuted:
                selected_points = self.irreducible_losses[:, global_index]
                permuted_points = selected_points[target, category]
                
                irreducible_losses.append(permuted_points)
                
            elif self.entropy:
                                
                #Return H[p(y|x,D_ho^{(c)})] -> (|C|, |N|, |Y|)
                logits = self.logits[i][global_index]
                print('logits shape', logits.shape)
                
                print('logist expsum', logits.logsumexp(dim=-1)[:, None].shape)
                
                normalised_prob = (logits - logits.logsumexp(dim=-1)[:,None]).exp()
                print('normalised prob shape', normalised_prob.shape)
                
                entropy = - (normalised_prob * logits).sum(axis=-1)
                print('entropy shape', entropy.shape)
                
                irreducible_losses.append(entropy)
            
            else:
                
                irreducible_losses.append(irred_loss_gen[global_index])
                                                
        return torch.stack(irreducible_losses)
    
    def assert_device(self, target_device:torch.device) -> None:
        
        #Move precomputed tensors to the target device:
        #for i, irred_loss in enumerate(self.irreducible_losses):
        #    if irred_loss.device != target_device:
        #        self.irreducible_losses[i] = self.irreducible_losses[i].to(device=target_device)
        if self.irreducible_losses.device != target_device:
            self.irreducible_losses = self.irreducible_losses.to(device=target_device)

        if self.entropy:
            for i, logits in enumerate(self.logits):
                if logits.device != logits:
                    self.logits[i] = self.logits[i].to(device=target_device)
    
    def check_precomputed_irreducible_losses(self, datamodule_config:dict, test:bool=False) -> None:
        
        #Setup temporary dataset and check the precomputed irreducible losses are handled correctly:
        datamodule_temp = hydra.utils.instantiate(datamodule_config)
        
        #Setup temporary sampler factory that doesn't use weights
        sampler_factory = ImbalancedSamplerFactory(num_classes=datamodule_temp.num_classes)
        datamodule_temp.setup(sampler_factory=sampler_factory, stage=datamodule_config.get('stage', None))
        
        #verify the correct dataset order: -> assertions in the verify method:
        log.info('Checking precomputed irreducible losses')
        
        if not test:
            
            num_irred_losses = len(self.irreducible_loss_generator)    
            for i, irred_loss_generator in enumerate(self.irreducible_loss_generator):
                
                log.info(f'Checking precomputed irreducible losses {i+1} of {num_irred_losses}')
                
                verify_correct_dataset_order(
                    dataloader=datamodule_temp.train_dataloader(),
                    sorted_target=irred_loss_generator["sorted_targets"],
                    idx_of_control_images=irred_loss_generator["idx_of_control_images"],
                    control_images=irred_loss_generator["control_images"],
                    dont_compare_control_images=datamodule_config.get(
                        "trainset_data_aug", False)) #Turn off dataset aug
        
        del datamodule_temp
        log.info('Finished checking precomputed irreducible losses')
        
        self.checked = True
    
    @abstractmethod
    def setup_irreducible_loss_generator(self):
        """
        Implements how the loader loads the multiple models into the precomputed
        model lists

        Returns
        -------
        None.

        """
        pass
                    
        
class PrecomputedReducibleLossGenerator(PrecomputedIrreducibleLossGenerator):
    
    def __init__(self, loader:LocalIO) -> None:
        
        self.loader = loader
        self.irreducible_loss_generator = loader.\
            load_losses_and_checks(irreducible_model=True)
        self.irreducible_losses = self.irreducible_loss_generator['irreducible_losses']
            
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        #Return the relevant precomputed values:
        return self.irreducible_losses[global_index]
    
    def assert_device(self, target_device:torch.device) -> None:
        
        #Move precomputed tensors to the target device:
        if self.irreducible_losses.device != target_device:
            self.irreducible_losses = self.irreducible_losses.to(device=target_device)
                
    def check_precomputed_irreducible_losses(self, datamodule_config:dict) -> None:
        
        #Setup temporary dataset and check the precomputed irreducible losses are handled correctly:
        datamodule_temp = hydra.utils.instantiate(datamodule_config)
        
        #Setup temporary sampler factory that doesn't use weights
        sampler_factory = ImbalancedSamplerFactory(num_classes=datamodule_temp.num_classes)
        datamodule_temp.setup(sampler_factory=sampler_factory, stage=datamodule_config.get('stage', None))
        
        #verify the correct dataset order: -> assertions in the verify method:
        verify_correct_dataset_order(
            dataloader=datamodule_temp.train_dataloader(),
            sorted_target=self.irreducible_loss_generator["sorted_targets"],
            idx_of_control_images=self.irreducible_loss_generator["idx_of_control_images"],
            control_images=self.irreducible_loss_generator["control_images"],
            dont_compare_control_images=datamodule_config.get(
                "trainset_data_aug", False)) #Turn off dataset aug
        
        del datamodule_temp

        
class PrecomputedWeightedReducibleLossGenerator(PrecomputedMultiIrreducibleLossGenerator):
    
    def __init__(self, 
                 loader:LocalIO, 
                 weights: ListConfig,
                 restricted: bool,
                 restricted_type: str,
                 permuted: bool,
                 entropy: bool) -> None:
        
        assert permuted == False,\
            'Permuted does not work with PrecomputedWeightedReducibleLossGenerator'
        
        super().__init__(loader=loader, restricted=restricted, 
                         restricted_type=restricted_type, permuted=permuted, 
                         entropy=entropy)
        
        assert weights is not None,\
            'PrecomputedWeightedReducibleLossGenerator must receive non None weights' 
                
        #Convert ListConfig into standard python list:    
        weights = OmegaConf.to_object(weights)
        self.weights = np.array(weights)
            
        #setup the initial irreducible loss generator
        self.setup_irreducible_loss_generator()
    
            
    def setup_irreducible_loss_generator(self):
        
        #Find index of non-zero weights to load relevant models:
        indices = np.arange(len(self.weights))
        non_zero_filter = self.weights > 0        
        self.non_zero_indices = indices[non_zero_filter]
        
        #Load relevant models:        
        log.info('Loading precomputed irreducible losses')
        
        for i, category in enumerate(self.non_zero_indices):
            
            log.info(f'Loading precomputed irreducible losses {i+1} of {len(self.non_zero_indices)}')
            
            irred_loss_generator = self.loader.\
                load_losses_and_checks(irreducible_model=True, class_imbalance=category)
    
            # Append relevant info to relevant structures:
            self.irreducible_loss_generator.append(irred_loss_generator)    
            self.irreducible_losses.append(irred_loss_generator['irreducible_losses'])
            
            if self.entropy:
                self.logits.append(irred_loss_generator['logits'])
            
                
        log.info('Finished loading precomputed irreducible losses')
        self.irreducible_losses = torch.stack(self.irreducible_losses)
                
class PrecomputedRobustReducibleLossGenerator(PrecomputedMultiIrreducibleLossGenerator):
    
    """
    Todo: Make Robust and WeightedReducibleLossGenerators inherit from the same base
    class with the same implementation of the majority of these functions
    """
    
    
    def __init__(self, 
                 loader:LocalIO, 
                 num_categories: int,
                 restricted: bool,
                 restricted_type: str,
                 permuted: bool,
                 entropy: bool) -> None:
                
        super().__init__(loader=loader, 
                         restricted=restricted,
                         restricted_type=restricted_type,
                         permuted=permuted,
                         entropy=entropy)

        #setup relevant variables:
        self.num_categories = num_categories
        
        #Every categories has a non_zero_index:
        self.non_zero_indices = np.arange(num_categories)
                
        #setup the initial irreducible loss generator
        self.setup_irreducible_loss_generator()
        
        #checking flag
        self.checked = False
                    
    def setup_irreducible_loss_generator(self):
                
        #Load relevant models:        
        log.info('Loading precomputed irreducible losses')
        
        for category in range(self.num_categories):
            
            log.info(f'Loading precomputed irreducible losses {category+1} of {self.num_categories}')
            
            irred_loss_generator = self.loader.\
                load_losses_and_checks(irreducible_model=True, class_imbalance=category)
    
            # Append relevant info to relevant structures:
            self.irreducible_loss_generator.append(irred_loss_generator)    
            self.irreducible_losses.append(irred_loss_generator['irreducible_losses'])
            
            #If using the entropy save the logits:
            if self.entropy:
                self.logits.append(irred_loss_generator['logits'])
                
        log.info('Finished loading precomputed irreducible losses')
        self.irreducible_losses = torch.stack(self.irreducible_losses)
        
class PrecomputedRobustMHReducibleLossGenerator(PrecomputedMultiIrreducibleLossGenerator):
    
    def __init__(self, 
                 loader:LocalIO, 
                 num_categories: int,
                 restricted: bool,
                 restricted_type: str,
                 permuted: bool,
                 entropy: bool) -> None:
                
        super().__init__(loader=loader, 
                         restricted=restricted,
                         restricted_type=restricted_type,
                         permuted=permuted,
                         entropy=entropy)
    
        #setup relevant variables:
        self.num_categories = num_categories
        
        #Every categories has a non_zero_index:
        self.non_zero_indices = np.arange(num_categories)
                
        #setup the initial irreducible loss generator
        self.setup_irreducible_loss_generator()
        
        #checking flag
        self.checked = False
        
    def setup_irreducible_loss_generator(self):
                
        #Load relevant models:        
        log.info('Loading multi headed precomputed irreducible losses')
    
        irred_loss_generator = self.loader.\
            load_losses_and_checks(irreducible_model=True) #class_imbalance specified in io_model
    
        self.irreducible_loss_generator = irred_loss_generator
        self.irreducible_losses = [ irred_loss['irreducible_losses'] for irred_loss in irred_loss_generator]
        
        #If using the entropy save the logits:
        if self.entropy:
            self.logits = [irred_loss['logits'] for irred_loss in irred_loss_generator]
                
        log.info('Finished loading precomputed irreducible losses')
        self.irreducible_losses = torch.stack(self.irreducible_losses)
            
class UpdatingIrreducibleLossGenerator(IrreducibleLossGenerator, ABC):
    
    @abstractmethod
    def config_optimizers(self, optim_config) -> None:
        pass
    
    @abstractmethod
    def gradient_descent_step(self, data:torch.Tensor, target:torch.Tensor,
                              global_index:torch.Tensor, category:torch.Tensor) -> None:
        pass
    
        
class UpdateIrreducibleReducibleLossGenerator(UpdatingIrreducibleLossGenerator):
    
    def __init__(self, loader:IO) -> None:
        
        self.loader = loader
        self.irreducible_loss_generator = loader.load_checkpoint(irreducible_model=True) 
        self.opt_irreducible_model = None
        self.loss = None
    
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        #Might want to add the choice of loss function to this setup:
        return F.cross_entropy(self.irreducible_loss_generator(data),
                               target, reduction="none")
    
    def assert_device(self, target_device) -> None:
        
        #Check if the device is the same:
        if self.irreducible_loss_generator.device != target_device:
            self.irreducible_loss_generator =\
                self.irreducible_loss_generator.to(device=target_device)
                
    def config_optimizers(self, optim_config:DictConfig, loss) -> None: #typing callable
        
        self.opt_irreducible_model = hydra.utils.instantiate(
                config=unmask_config(optim_config),
                params=self.irreducible_loss_generator.parameters(),
                _convert_="partial",
            )
        
        #Not sure what's happening here?
        for param_group in self.opt_irreducible_model.param_groups:
            param_group['lr'] = param_group['lr']/100
            
        self.loss = loss
            
    def gradient_descent_step(self, data:torch.Tensor, target:torch.Tensor,
                              global_index:torch.Tensor, category:torch.Tensor) -> list:
        
        # ensure config_optimizers have been run first:
        assert self.opt_irreducible_model is not None,\
            'opt_irreducible_model is None, must run config_optimizers first'
        assert self.loss is not None,\
            'loss must not be None'
        
        # Look into various enable grad flags:
        self.opt_irreducible_model.zero_grad()
        logits = self.irreducible_loss_generator(data)
        irreducible_loss = self.loss(logits, target).mean() #we need the model loss:
        irreducible_loss.backward()
        self.opt_irreducible_model.step()
        
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        irreducible_acc = accuracy(preds, target)
        
        return [(irreducible_loss.detach().cpu(), irreducible_acc)]
    
    def calculate_losses_and_accuracy(self, data:torch.Tensor, target:torch.Tensor,
                              global_index:torch.Tensor, category:torch.Tensor) -> list:
        
        self.assert_device(target_device=data.device)
        
        #Calculate the losses and accuracy of the model:
        logits = self.irreducible_loss_generator(data)
        irlomo_loss = self.loss(logits, target)
        preds = torch.argmax(logits, dim=1)
        irlomo_acc = accuracy(preds, target)
        
        return [(irlomo_loss.mean(), irlomo_acc)]
    
class UpdatingMultiIrreducibleLossGenerator(IrreducibleLossGenerator, ABC):
    
    @abstractmethod
    def setup_irreducible_loss_generator(self):
        pass
        
class UpdateIrreducibleWeightedReducibleLossGenerator(UpdatingIrreducibleLossGenerator):
    
    def __init__(self, loader:IO,
                 weights: ListConfig) -> None:
        
        assert weights is not None,\
            'PrecomputedWeightedReducibleLossGenerator must receive non None weights' 
                
        #Convert ListConfig into standard python list:    
        weights = OmegaConf.to_object(weights)
        self.weights = np.array(weights)  
        

        self.loader = loader

        # Init key variables:        
        self.irreducible_loss_generators = list()
        self.opt_irreducible_models = list()
        self.loss = None

        # Setup irreducible loss generator:        
        self.setup_irreducible_loss_generator()
    
    
    def setup_irreducible_loss_generator(self):
        
        #Find index of non-zero weights to load relevant models:
        indices = np.arange(len(self.weights))
        non_zero_filter = self.weights > 0        
        self._non_zero_indices = indices[non_zero_filter]
        
        #Load relevant models:        
        log.info('Loading irreducible loss checkpoints')
        
        for i, category in enumerate(self._non_zero_indices):
            
            log.info(f'Loading irreducible loss checkpoints {i+1} of {len(self.non_zero_indices)}')
            
            irred_loss_generator = self.loader.\
                load_checkpoint(irreducible_model=True, class_imbalance=category)
    
            # Append relevant info to relevant structures:
            self.irreducible_loss_generators.append(irred_loss_generator)    
                
        log.info('Finished loading irreducible loss checkpoints')
        
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        output = list()
        
        #Return irreducible losses matrix:
        for i, irred_loss_model in enumerate(self.irreducible_loss_generators):
            
            output.append(F.cross_entropy(irred_loss_model(data),
                               target, reduction="none"))
    
        return torch.stack(output)
    
    def assert_device(self, target_device) -> None:
                
        for i, irred_loss in enumerate(self.irreducible_loss_generators):
            if irred_loss.device != target_device:
                self.irreducible_loss_generators[i] = \
                    self.irreducible_loss_generators[i].to(device=target_device)
                        
    def config_optimizers(self, optim_config:DictConfig, loss) -> None: #typing callable
        
        assert len(self.irreducible_loss_generators) > 0,\
            'Must run setup_irreducible_loss_generator before configuring the opt'
    
        for i, _ in enumerate(self.irreducible_loss_generators):
    
            self.opt_irreducible_models.append(hydra.utils.instantiate(
                                config=unmask_config(optim_config),
                                params=self.irreducible_loss_generators[i].parameters(),
                                _convert_="partial"))
        
            #Not sure what's happening here?
            for param_group in self.opt_irreducible_models[i].param_groups:
                param_group['lr'] = param_group['lr']/100
            
        self.loss = loss
            
    def gradient_descent_step(self, data:torch.Tensor, target:torch.Tensor,
                              global_index:torch.Tensor, category:torch.Tensor,
                              filter_category:bool=False) -> list:
        
        # ensure config_optimizers have been run first:
        assert len(self.opt_irreducible_models) > 0,\
            'opt_irreducible_model is None, must run config_optimizers first'
        assert self.loss is not None,\
            'loss must not be None'
          
        log_outputs = list()   
     
        for i, _ in enumerate(self.irreducible_loss_generators):  
            
            if filter_category: #restrict grad update to only terms in category
                category_filter = category == self._non_zero_indices[i]
            else:
                category_filter = category == category
                
            # Look into various enable grad flags:
            self.opt_irreducible_models[i].zero_grad()
            logits = self.irreducible_loss_generators[i](data[category_filter])
            irreducible_loss = self.loss(logits,
                                         target[category_filter]).mean() #we need the model loss:
            irreducible_loss.backward()
            self.opt_irreducible_models[i].step()
            
            #Calculate logged variables:
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            irreducible_acc = accuracy(preds, target[category_filter])
            log_outputs.append((irreducible_loss.detach().cpu(),
                                irreducible_acc))
            
        return log_outputs
    
    def calculate_losses_and_accuracy(self, data:torch.Tensor, target:torch.Tensor,
                              global_index:torch.Tensor, category:torch.Tensor) -> list:
        
        log_output = list()
        self.assert_device(target_device=data.device) #What's happening here...
        
        for i, irred_loss_model in enumerate(self.irreducible_loss_generators):
            
            #Calculate the losses and accuracy of the model:
            logits = irred_loss_model(data)
            irlomo_loss = self.loss(logits, target)
            preds = torch.argmax(logits, dim=1)
            irlomo_acc = accuracy(preds, target)
        
            log_output.append((irlomo_loss.mean(), irlomo_acc))
        
        return log_output
    

class ModelIrreducibleLossGenerator(IrreducibleLossGenerator):
    
    def __init__(self, loader:IO) -> None:
        
        self.loader = loader
        self.irreducible_loss_generator = loader.load_checkpoint(irreducible_model=True) 
        self.opt_irreducible_model = None
        self.loss = None
    
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        with torch.no_grad():
        
            #Might want to add the choice of loss function to this setup:
            return F.cross_entropy(self.irreducible_loss_generator(data),
                                   target, reduction="none")
    
    def assert_device(self, target_device) -> None:
        
        #Check if the device is the same:
        if self.irreducible_loss_generator.device != target_device:
            self.irreducible_loss_generator =\
                self.irreducible_loss_generator.to(device=target_device)
                                

class MultiModelIrreducibleLossGenerator(IrreducibleLossGenerator):
    
    def __init__(self, loader:IO, num_categories:int) -> None:
        
        #Convert ListConfig into standard python list:            
        self.loader = loader
        self.num_categories = num_categories

        # Init key variables:        
        self.irreducible_loss_generators = list()
        self.opt_irreducible_models = list()
        self.loss = None

        # Setup irreducible loss generator:        
        self.setup_irreducible_loss_generator()
    
    
    def setup_irreducible_loss_generator(self):
                
        #Load relevant models:        
        log.info('Loading irreducible loss checkpoints')
        
        for i, category in enumerate(range(self.num_categories)):
            
            log.info(f'Loading irreducible loss checkpoints {i+1} of {self.num_categories}')
            
            irred_loss_generator = self.loader.\
                load_checkpoint(irreducible_model=True, class_imbalance=category)
    
            # Append relevant info to relevant structures:
            self.irreducible_loss_generators.append(irred_loss_generator)    
                
        log.info('Finished loading irreducible loss checkpoints')
        
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        output = list()
        
        #Return irreducible losses matrix:
        for i, irred_loss_model in enumerate(self.irreducible_loss_generators):
            with torch.no_grad():
            
                output.append(F.cross_entropy(irred_loss_model(data),
                                   target, reduction="none"))
    
        return torch.stack(output)
    
    def assert_device(self, target_device) -> None:
                
        for i, irred_loss in enumerate(self.irreducible_loss_generators):
            if irred_loss.device != target_device:
                self.irreducible_loss_generators[i] = \
                    self.irreducible_loss_generators[i].to(device=target_device)
                    
class MultiHeadedModelIrreducibleLossGenerator(IrreducibleLossGenerator):
    
    def __init__(self, loader:IO, num_categories:int) -> None:
        
        #Convert ListConfig into standard python list:            
        self.loader = loader
        self.num_categories = num_categories

        # Init key variables:        
        self.irreducible_loss_generators = list()
        self.opt_irreducible_models = list()
        self.loss = None

        # Setup irreducible loss generator:        
        self.setup_irreducible_loss_generator()
    
    
    def setup_irreducible_loss_generator(self):
                
        #Load relevant models:        
        log.info('Loading irreducible loss checkpoints')
        
        self.irred_loss_generator = self.loader.load_checkpoint(irreducible_model=True, 
                                                                class_imbalance=None) #Get this right
                        
        log.info('Finished loading irreducible loss checkpoints')
        
    def calculate_irreducible_losses(self, global_index:torch.Tensor,
                                     data:torch.Tensor, 
                                     target:torch.Tensor, 
                                     category:torch.Tensor) -> torch.Tensor:
        
        output = list()
        logits = self.irred_loss_generator(data) 
        
        for c in range(self.num_categories):
                
            output.append(F.cross_entropy(logits[c], target, reduction="none"))
    
        return torch.stack(output)
    
    def assert_device(self, target_device) -> None:
                
        if self.irred_loss_generator.device != target_device:
            self.irred_loss_generator = self.irred_loss_generators.to(device=target_device)
                        