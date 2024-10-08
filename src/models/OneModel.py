# regular imports
import hydra

from typing import Union

# lightning related imports
import pytorch_lightning as pl

# pytorch related imports
import torch
import numpy as np
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F

from src.datamodules.datamodules import (
    CINIC10RelevanceDataModule, 
    Clothing1MDataModule,
    CIFAR100DataModule)
from src.utils.utils import (
    unmask_config,
    process_batch)


class OneModel(pl.LightningModule):
    def __init__(
        self,
        model,
        optimizer_config,
        scheduler_config,
        model_io,
        datamodule=None,
        number_training_steps:int=None,
        percent_train=1.0,
        number_of_classes:Union[None, int]=None,
        number_of_groups: Union[None, int]=None,
        gradient_weighted_class:Union[None, int]=None,
        gradient_weight:Union[None, float]=None,
        scheduler_step_on_step=False,
        loaded_from_checkpoint = True,
        multi_headed_model = False,
    ):
        """
        

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        optimizer_config : TYPE
            DESCRIPTION.
        scheduler_config : TYPE
            DESCRIPTION.
        model_io : TYPE
            DESCRIPTION.
        datamodule : TYPE, optional
            DESCRIPTION. The default is None.
        percent_train : TYPE, optional
            DESCRIPTION. The default is 1.0.
        gradient_weighted_class : Union[None, int], optional
            DESCRIPTION. The default is None.
        gradient_weight : Union[None, float], optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        
        super().__init__()

        self.model_io = model_io

        # log hyperparameters
        self.save_hyperparameters()

        self.model = model
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.datamodule = datamodule
        
        self.number_training_steps = number_training_steps
        self.scheduler_step_on_step = scheduler_step_on_step
        self.number_of_classes = number_of_classes
        self.number_of_groups = number_of_groups
        
        #TODO: Need to set this up better when we move to non-label categories
        self.validation_acc_per_label = np.zeros(number_of_classes)
        self.validation_points_per_label = np.zeros(number_of_classes)
        
        #Gradient weighting
        self.gradient_weighted_class=gradient_weighted_class
        self.gradient_weight=gradient_weight
        self.multi_headed_model = multi_headed_model

        if self.gradient_weighted_class is not None:
            assert self.multi_headed_model is False,\
                'gradient_weighted_class cannot be set to {self.gradient_weighted_class}, whilst multi_headed_model is not False'
            assert self.gradient_weight is not None,\
                "gradient_weight and gradient_weighted_class must both be None or not None"
                
        if self.gradient_weight is not None:
            assert (self.multi_headed_model is True) or (self.gradient_weighted_class is not None),\
                'if gradient_weight is not None either multi_headed_model must be True or gradient_weight_class must not be None'
            
        #Construct Weight vector:
        if self.gradient_weight is not None:
                        
            #For the CIFAR100 example we implement group weights
            if self.number_of_groups is not None:
                
                assert self.gradient_weighted_class < self.number_of_groups,\
                    f'gradient_weighted_class: {self.gradient_weighted_class} must be less than: {datamodule.num_groups}'
                
                #Setup to the number of classes either way
                weights = torch.ones(self.number_of_classes)
                
                if self.datamodule is not None: #Evaluate the exact weights
                    for key, value in datamodule.indices_train.dataset.group_index.items():
                        if value == self.gradient_weighted_class:
                            weights[key] = self.gradient_weight
            
            else:
                weights = torch.ones(number_of_classes)
                weights[self.gradient_weighted_class] = self.gradient_weight

            self.loss = nn.CrossEntropyLoss(weight=weights)
        
        else:
            self.loss = nn.CrossEntropyLoss()
             
        

    def forward(self, x):
        x = self.model(x)
        return x
    
    def multi_head_step(self, logits, target):
        
        assert isinstance(logits, list),\
            f"When multi_headed_model True, forward output expected to return type list not {type(logits)}"
        assert self.gradient_weight is not None, 'When multi headed model True, gradient_weight must be set'
        
        loss = torch.Tensor([0]).to(target.device)
        for c in range(self.number_of_classes):
            
            #Account for the case when the entire batch is one class or another:
            target_filter = target == c
            if target_filter.sum() == len(logits[0]):
                loss +=  self.loss(logits[c][target_filter], target[target_filter]) * self.gradient_weight
            elif target_filter.sum() == 0:
                loss +=  self.loss(logits[c][~target_filter], target[~target_filter])
            else:
                loss +=  self.loss(logits[c][target_filter], target[target_filter]) * self.gradient_weight
                loss +=  self.loss(logits[c][~target_filter], target[~target_filter])
        
        return loss
    
    def multi_head_logging(self, logits, target, on_epoch:bool, on_step:bool, split:str='train'):
        
        losses = list()
        for c in range(self.number_of_classes):
            
              loss = torch.Tensor([0]).to(target.device)
                         
              target_filter = target == c
              if target_filter.sum() == len(logits[0]):
                  loss +=  self.loss(logits[c][target_filter], target[target_filter]) * self.gradient_weight
              elif target_filter.sum() == 0:
                  loss +=  self.loss(logits[c][~target_filter], target[~target_filter])
              else:
                  loss +=  self.loss(logits[c][target_filter], target[target_filter]) * self.gradient_weight
                  loss +=  self.loss(logits[c][~target_filter], target[~target_filter])
            
              losses.append(loss.detach().cpu())
              self.log(f"{split}_loss_head_{c}_class{c}", loss, on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
    
    
        if not on_step:
            val_name = f'{split}_loss_epoch'
            self.log(val_name, np.mean(losses), on_epoch=on_epoch, logger=True, sync_dist=True)
        
        else:
            val_name = f'{split}_loss'
            self.log(val_name, np.mean(losses), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
        
        return np.mean(losses)
    

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        
        _, data, target, _ = process_batch(batch)       

        batch_size = len(target)
        
        if self.hparams.percent_train < 1:
            selected_batch_size = int(batch_size * self.hparams.percent_train)
            selected_minibatch = torch.randperm(len(data))[:selected_batch_size]
            data = data[selected_minibatch]
            target = target[selected_minibatch]

        logits = self.model(data)
        
        #For a multi headed model calculate the loss
        if self.multi_headed_model:
            loss = self.multi_head_step(logits, target)
            self.multi_head_logging(logits, target, True, True, split='train')
        else:
            loss = self.loss(logits, target)

            # training metrics
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            acc = accuracy(preds, target, task="multiclass", num_classes=self.number_of_classes)
            self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        #Add scheduler step at a step level for the NLP models
        if (self.scheduler_config is not None) and \
            (self.scheduler_step_on_step):
            
            scheduler = self.lr_schedulers()
            scheduler.step()
            
            self.log(
                'lr_rate', scheduler.get_lr()[0], on_step=True, on_epoch=True, logger=True
                )

        if isinstance(self.datamodule, CINIC10RelevanceDataModule):
            self.log(
                "percentage_relevant",
                self.datamodule.percentage_targets_relevant(target),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
                        
        _, data, target, category = process_batch(batch)
        
        logits = self.model(data)
        
        if self.multi_headed_model:
            loss = self.multi_head_logging(logits, target, on_epoch=True, on_step=False, split='val')
        else:
            loss = self.loss(logits, target)

            # training metrics
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            acc = accuracy(preds, target, task="multiclass", num_classes=self.number_of_classes)
            self.log("val_loss_epoch", loss, on_epoch=True, logger=True, sync_dist=True)
            self.log("val_acc_epoch", acc, on_epoch=True, logger=True, sync_dist=True)
                

            #Calculate the class specific validation loss
            for c in range(self.number_of_classes):
                            
                #Throw error if no instances of class in the data: 
                try:
                    
                    class_loss = self.loss(logits[category==c], target[category==c])
                    
                    self.log(f'class_{c}_val_loss',
                             class_loss,
                             on_step=True,
                             on_epoch=True,
                             logger=True)
                    
                    class_acc = accuracy(preds[target==c],
                                         target[target==c],
                                         task="multiclass",
                                         num_classes=self.number_of_classes)
                    
                    self.log(f'class_{c}_val_acc',
                             class_acc,
                             on_step=True,
                             on_epoch=True,
                             logger=True)
                    
                except RuntimeError as e:
                    #RuntimeError thrown when len(preds[target==i]) == 0
                    assert len(preds[target==c]==0) or len(target[target==c])==0,\
                        f"{e} thrown when neither preds or target have length zero"
                                            
                except IndexError as e:
                    assert len(preds[target==c]==0) or len(target[target==c])==0,\
                        f"{e} thrown when neither preds or target have length zero" 

        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        
        _, data, target, category = process_batch(batch)
        logits = self.model(data)
        
        if self.multi_headed_model:
            loss = self.multi_head_logging(logits, target, on_epoch=True, on_step=False, split='test')
        else:
        
            loss = self.loss(logits, target)
    
            # training metrics
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            acc = accuracy(preds, target, task="multiclass", num_classes=self.number_of_classes)
            self.log("test_loss_epoch", loss, on_epoch=True, logger=True)
            self.log("test_acc_epoch", acc, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            config=unmask_config(self.optimizer_config),
            params=self.model.parameters(),
            _convert_="partial",
        )

        if self.scheduler_config is None:
            return [optimizer]
        else:
            
            #If total iters flag exists set it to the number of training steps:
            if self.scheduler_config.get('total_iters', None) is not None:
                
                print(self.scheduler_config)
                print(self.scheduler_config['total_iters'])
                self.scheduler_config['total_iters'] = self.number_training_steps                    
                print(self.scheduler_config['total_iters'])
            
            
            scheduler = hydra.utils.instantiate(
                unmask_config(self.scheduler_config),
                optimizer=optimizer,
                _convert_="partial",
            )
            return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        
        #Change the datamodule and model_io if they point to local paths
        if checkpoint['hyper_parameters'].get('datamodule', None) is not None:
            checkpoint['hyper_parameters']['datamodule'] = None            
        
        #Save the model checkpoint using the model_io
        self.model_io.save_checkpoint(checkpoint, irreducible_model=True)
   
if __name__ == '__main__':
    
    import os 
    
    path = r'C:\Users\William\Documents\Programming\PhD\Datasets\Robust_RHO_Project\MNLIDataModule\irreducible_models\irred_model_checkpoint_100.ckpt'
    
    loaded = torch.load(path)
        
    #model = OneModel.load_from_checkpoint(checkpoint_path=path, strict=True)
        