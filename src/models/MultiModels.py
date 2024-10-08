# regular imports
import math
import numbers
import time

import numpy as np

# lightning related imports
import pytorch_lightning as pl

# pytorch related imports
import torch
from torchmetrics.functional import accuracy
from torch import nn
from torch.nn import functional as F
import pandas as pd
from omegaconf import open_dict

from transformers.modeling_outputs import SequenceClassifierOutput

from src.curricula.utils_bald import enable_dropout
from src.utils.utils import (
    process_batch,
    unmask_config)

import hydra
import wandb

from src.datamodules.datamodules import CINIC10RelevanceDataModule, Clothing1MDataModule
from src.curricula.selection_methods import (
    reducible_loss_selection, 
    irreducible_loss_selection,
    gradnorm_ub_selection,
    ce_loss_selection,
    uniform_selection,
    robust_reducible_loss_selection,
    weighted_reducible_loss_selection,
    robust_payoff_reducible_loss_selection)


class MultiModels(pl.LightningModule):
    def __init__(
        self,
        large_model,
        irreducible_loss_generator=None,
        proxy_model=None,
        selection_method=None,
        optimizer_config=None,
        scheduler_config=None,
        scheduler_step_on_step=False,
        number_training_steps=None,
        model_io=None,
        percent_train=0.1,
        detailed_logging=False,
        num_mc=10,  # number of MC samples if using BALD
        datamodule=None,
        num_classes=None,
        num_groups=None,
        update_irreducible=False,
        repetition_logging=False,
        parallel_implementation=False,
        parallel_skip=False,
        selection_train_mode=True,
        track_all_selection=False,
        track_diversity=False,
        run_test_eval=False,
        robust_holdout_loss_sum=False,
        seed=None
    ):
        """
        PyTorch Lightning Module for GoldiProx.
        Args:
            large_model: nn.Module, large model in goldiprox setting
            irreducible_loss_generator: Tensor or nn.Module
             Tensor: with irreducible losses for train set, ordered by <index> (see datamodules)
             nn.Module: irreducible loss model
            proxy_model: nn.Module, a model that acts as proxy for large_model
            selection_method: callable class, selection method. current available options include: reducible loss, irreducible loss, ce_loss, uniform, bald
            learning_rate: float, learning rate for all models
            percent_train: float [0-1], the percent of each batch to train on
            update_irreducible: bool, update irreducible loss model with respect to training data
            detailed_logging: bool, detailed loggin
            repetition_logging: bool, enables loging how many of the points are repeated
            selection_train_mode: bool; whether the selection is done with the
                model in .train mode or .eval mode. This influences batch norm
                behaviour and dropout layers. Defaults to True.
        """
        super().__init__()

        # turn off PL's automatic optimisation so we can optimise per GoldiProx algorithm
        self.automatic_optimization = False

        # log and save hyperparameters
        self.save_hyperparameters()
        # saved to self.hparams
        # self.detailed_logging = detailed_logging
        # self.learning_rate = learning_rate
        # self.percent_train = percent_train
        
        # should be defined or instantiated by hydra
        self.selection_method = selection_method
        self.large_model = large_model
        self.proxy_model = proxy_model
        self.irreducible_loss_generator = irreducible_loss_generator
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_step_on_step = scheduler_step_on_step
        self.number_training_steps = number_training_steps
        self.model_io = model_io
        self.datamodule = datamodule
        self.track_diversity = track_diversity
        self.run_test_eval = run_test_eval
        self.robust_holdout_loss_sum = robust_holdout_loss_sum
        self.seed=seed

        self.update_irreducible = update_irreducible

        # loss function
        self.loss = nn.CrossEntropyLoss(reduction="none")

        # For recording sequence used in training
        self.sequence = np.asarray([])

        self.detailed_log = []
        self.repetition_logging = repetition_logging

        # store stale gradients here. Only used if parallel_implementation is true
        self.saved_batch = None
        self.current_batch = None
        
        self.num_classes = num_classes
        self.num_groups = num_groups
        
        #Get size of labels
        if num_groups is None:
            num_groups = num_classes
            self.num_groups = num_classes
        
        if num_groups is not None:
            self.labels_selected_per_epoch = np.zeros(num_groups)

            #TODO: Need to set this up better when we move to non-label categories
            self.validation_acc_per_label = np.zeros(num_groups)
            self.validation_points_per_label = np.zeros(num_groups)
            
            self.training_loss_per_label = np.zeros(num_groups)
            self.training_acc_per_label = np.zeros(num_groups)
            self.training_points_per_label = np.zeros(num_groups)
            
        if (isinstance(selection_method, robust_reducible_loss_selection)) or \
            (isinstance(selection_method, weighted_reducible_loss_selection)) or \
            (isinstance(selection_method, robust_payoff_reducible_loss_selection)):
            assert num_groups is not None,\
                'size_of_labels arg must be given when using robust_reducible_loss_selection'
            
            self.num_points_in_class = torch.zeros(num_groups).cpu()
            self.running_sum_of_robust_holdout_losses = torch.zeros(num_groups).cpu()
            self.robust_holdout_loss = torch.zeros(num_groups).cpu()
        else:
            self.robust_holdout_loss = None
          
        if (datamodule is not None) and (self.track_diversity is not False): 
            max_index = self.datamodule._get_set_of_global_indices()
            self.selected_index_counts = np.zeros(max_index + 1)
            self.target_index_counts = np.zeros(max_index + 1) - 1

        if track_all_selection:
            self.all_selection_methods = [
                reducible_loss_selection(),
                gradnorm_ub_selection(),
                ce_loss_selection(),
                irreducible_loss_selection(),
                uniform_selection()
                ]
            self.all_selection_method_names = [
                "redloss",
                "gradnorm",
                "ce_loss",
                "irred_loss",
                "uniform"
                ]

    def on_fit_start(self):
        pl.seed_everything(self.seed, workers=True)

    def forward(self, x):
        x = self.large_model(x)
        return x
    
    def training_step(self, batch, batch_idx):
         
        
        global_index, data, target, categories = process_batch(batch)
        batch_size = len(target)
        
        # #For dictionary NLP data
        # if isinstance(batch, dict):
        #     global_index = batch.pop("idx")
        #     inputs = batch
        #     data = inputs
        #     target = inputs["labels"]
        #     batch_size = len(target)
        #     categories = batch.pop("categories")
        
        # else:
        
        #     global_index, data, target, categories = batch
        #     batch_size = len(data)
            
        #Calculate the selected batch size:
        selected_batch_size = max(1, int(batch_size * self.hparams.percent_train))
        
        #Time each iteration:
        start_time = time.time()

        if self.hparams.selection_train_mode:
            self.large_model.train()
        else:
            self.large_model.eval() # switch to eval mode to compute selection
        ### Selection Methods
        selected_indices, metrics_to_log, irreducible_loss = self.selection_method(
            selected_batch_size=selected_batch_size,
            data=data,
            target=target,
            categories=categories,
            global_index=global_index,
            large_model=self.large_model,
            irreducible_loss_generator=self.irreducible_loss_generator,
            proxy_model=self.proxy_model,
            current_epoch=self.current_epoch,  # not used by all methods, but needed for annealing
            num_classes=self.datamodule.num_classes,
            robust_holdout_loss=self.robust_holdout_loss
        )  # irreducible_loss will be None if the selection_method does not involve
        # irreducible_loss computation (e.g. uniform, CE loss selection)
        self.large_model.train()  # switch to eval mode to compute selection

        pc_corrupted = self.datamodule.percentage_corrupted(
            global_index[selected_indices]
        )
        if pc_corrupted:  # returns None if no corruption was applied
            self.log(
                "selected_percentage_corrupted",
                pc_corrupted,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        if isinstance(self.datamodule, Clothing1MDataModule) and self.hparams.track_all_selection:
            for name, selection_method in zip(self.all_selection_method_names, self.all_selection_methods):
                selected_indices_method, _, _ = selection_method(
                    selected_batch_size=selected_batch_size,
                    data=data,
                    target=target,
                    global_index=global_index,
                    large_model=self.large_model,
                    irreducible_loss_generator=self.irreducible_loss_generator,
                    proxy_model=self.proxy_model,
                    current_epoch=self.current_epoch,  # not used by all methods, but needed for annealing
                    num_classes=self.datamodule.num_classes
                    )
                self.log(
                    "percentage_clean_"+name,
                    self.datamodule.percentage_clean(global_index[selected_indices_method]),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )


        selection_time = time.time() - start_time
        self.log('selection_time', selection_time, on_step=True, on_epoch=True, logger=True)

        # build sequence
        self.sequence = np.append(
            self.sequence, global_index[selected_indices].cpu().numpy()
        )
        og_target = target #preserve the original batch targets for future logging
        target, categories = target[selected_indices],\
                            categories[selected_indices]
        #Change this for dictionary NLP datasets...
        if isinstance(data, dict):
            new_data = dict()
            for key, item in data.items():
                new_data[key] = item[selected_indices]
            
            data = new_data
        else:
            data = data[selected_indices]

        if self.hparams.parallel_implementation:
            self.current_batch = (data, target)

            # save the current selected batch
            if self.saved_batch is None:
                self.saved_batch = (
                    self.current_batch
                )  # for the first step, use the selected points from the first batch. reused in the next step

                if self.hparams.parallel_skip:
                    return  # skip step

            data, target = self.saved_batch  # load the stale batch
            self.saved_batch = self.current_batch  # save the current batch

        # repetition logging made optional because it requires no shuffling (and
        # also currently fails with CIFAR)
        if self.repetition_logging:
            self.log_repetition(data)

        if self.proxy_model is not None:
            opt_large_model, opt_proxy_model = self.optimizers()

            opt_proxy_model.zero_grad()
            logits = self.proxy_model(data)

            proxy_model_loss = self.loss(logits, target)
            self.manual_backward(proxy_model_loss.mean())
            opt_proxy_model.step()

            # logging
            preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
            proxy_model_acc = accuracy(preds, target, task="multiclass", num_classes=self.num_classes)
            self.log(
                "proxy_train_loss",
                proxy_model_loss.mean(),
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "proxy_train_acc",
                proxy_model_acc,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        elif self.hparams.update_irreducible:
            opt_large_model = self.optimizers()

            irred_loss_log_outputs = self.irreducible_loss_generator.\
                                                gradient_descent_step(
                                                    data=data,
                                                    target=target,
                                                    global_index=global_index,
                                                    category=categories)
            
            # We always treat irred_loss_log_outputs as a list:
            for i, (irred_loss_mean, irred_loss_acc) in enumerate(irred_loss_log_outputs):
                
                self.log(
                    f"IrLoMo_train_loss_{i}",
                    irred_loss_mean,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    f"IrLoMo_train_acc_{i}",
                    irred_loss_acc,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )


        else:
            opt_large_model = self.optimizers()

        #This is the actual training step

        opt_large_model.zero_grad()
        logits = self.large_model(data)
        
        if isinstance(logits, SequenceClassifierOutput):
            logits = logits[1]
        
        loss = self.loss(logits, target)
        self.manual_backward(loss.mean())
        opt_large_model.step()
        
        #Add scheduler step at a step level for the NLP models
        if (self.scheduler_config is not None) and \
            (self.scheduler_step_on_step):
            
            scheduler = self.lr_schedulers()
            scheduler.step()
            
            self.log(
                'lr_rate', scheduler.get_lr()[0], on_step=True, on_epoch=True, logger=True
                )
        
        # Finish time recording here after back prop
        full_train_step_time = time.time() - start_time
        self.log('training_step_time', full_train_step_time,
                 on_step=True, on_epoch=True, logger=True)
        
        # training metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target, task="multiclass", num_classes=self.num_classes)
        self.log("train_loss", loss.mean(), on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)

        if isinstance(self.datamodule, CINIC10RelevanceDataModule):
            self.log(
                "percentage_relevant",
                self.datamodule.percentage_targets_relevant(target),
                on_step=True,
                on_epoch=True,
                logger=True,
            )

        detailed_only_keys = metrics_to_log["detailed_only_keys"]
        metrics_to_log[
            "step"
        ] = (
            self.global_step
        )  # add step to the logging, also might help us concretely cross-corelate exact point in time.
            
        # batch statistics summary logging, depending on the metric that we ended up using.
        for key, value in metrics_to_log.items():
            if key in detailed_only_keys:
                continue

            if isinstance(value, np.ndarray):
                for percentile in [2.5, 25, 50, 75, 97.5]:
                    v = np.percentile(value, percentile)
                    self.log(
                        f"{key}_{percentile}",
                        v,
                        on_step=True,
                        on_epoch=True,
                        logger=True,
                    )

            elif isinstance(value, numbers.Number):
                self.log(key, value, on_step=True, on_epoch=True, logger=True)

        #Robust RHO selection specific logging:
        if 'losses' in metrics_to_log.keys():
            for i, loss_log in enumerate(metrics_to_log['losses']):
                self.log('losses_{}'.format(i),
                         loss_log,
                         on_step=True,
                         on_epoch=True,
                         logger=True)    
                
        #REFORMAT this code setup
        if 'weights' in metrics_to_log.keys():
            for i, weights in enumerate(metrics_to_log['weights']):
                self.log('weights_{}'.format(i),
                         weights,
                         on_step=True,
                         on_epoch=True,
                         logger=True)

        # unclear to me quite how inefficient this will be. We can use the lightning profiler :~)
        if self.hparams.detailed_logging:
            self.detailed_log.append(metrics_to_log)
            
        #Add rolling statistic across number of classes here!!!!
        #Rolling statistic to check no. of classes:   
        if self.num_groups is not None:
            
            #Torch bin count has some funky behaviour ... check in
            selected_labels_counts = torch.bincount(categories).cpu().numpy()
            max_selected_labels = categories.max().cpu().numpy()            
            self.labels_selected_per_epoch[:max_selected_labels+1] += selected_labels_counts        
                   
            #log labels selected per class per epoch: -> size of labels is a stupid name!
            for i, x in enumerate(self.labels_selected_per_epoch):
                self.log('labels_selected_per_epoch_{}'.format(i),
                         x,
                         on_step=True,
                         on_epoch=True,
                         logger=True)
                
                self.log('labels_selected_each_step_{}'.format(i),
                        (categories == i).sum().detach().cpu(),
                        on_step=True,
                        on_epoch=True,
                        logger=True)
                        
                #record the number of points selected per label:
                self.training_points_per_label[i] += (categories==i).sum().\
                                                    detach().cpu().numpy()
                
                #Calculate the per class training loss
                class_loss = loss[categories == i].sum()
                if not class_loss.isnan():
                    self.training_loss_per_label[i] += class_loss.detach().cpu().numpy()
                                                
                #Calculate the per class training accuracy
                #Catch errors thrown when no points from the class are included...
                try: 
                    class_acc = accuracy(preds[categories==i],
                                         target[categories==i], task="multiclass", num_classes=self.num_classes)
                    self.training_acc_per_label[i] += ((categories==i).sum() * class_acc).\
                                                        detach().cpu().numpy()
                        
                # Specifically - max(): Expected reduction dim to be specified for input.numel() == 0.
                except RuntimeError as e:
                    #RuntimeError thrown when len(preds[target==i]) == 0
                    assert len(preds[categories==i]==0) or len(categories[categories==i])==0,\
                        f"{e} thrown when neither preds or target have length zero"                   

        
        ######################################################################        
        #For the first epoch identify training epoch global indices:
        if self.track_diversity:
            _global_index = global_index.cpu().numpy()
            _selected_indices = selected_indices.cpu().numpy()
                
            #Label the target at every iteration s.t. even if the point isn't sampled early the labels update
            self.target_index_counts[_global_index] = og_target.cpu().numpy() 
            
            #For every step in every epoch update with rolling count of selected indices:
            self.selected_index_counts[_global_index[_selected_indices]] += 1
            #These stats are logged in the on_training_step_end method.        

        if self.proxy_model is not None:

            # track correlation and covariance between proxy_model and big model
            spearman = self.spearman_correlation(proxy_model_loss, loss)
            self.log(
                "spearman_proxy_loss_large_loss",
                spearman,
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            cov = np.cov(
                proxy_model_loss.detach().cpu().numpy(), loss.detach().cpu().numpy()
            )[0, 1]
            self.log(
                "cov_proxy_loss_large_loss",
                cov,
                on_step=True,
                on_epoch=True,
                logger=True,
            )

            # track standard deviations
            std = torch.std(proxy_model_loss)
            self.log("std_proxy_loss", std, on_step=True, on_epoch=True, logger=True)

            std = torch.std(loss)
            self.log("std_large_loss", std, on_step=True, on_epoch=True, logger=True)

            if irreducible_loss is not None:

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, loss - irreducible_loss
                )
                self.log(
                    "spearman_proxy_redloss_large_redloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                # Correlations between the same metrics on different models

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, loss - irreducible_loss
                )
                self.log(
                    "spearman_proxy_redloss_large_redloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                cov = np.cov(
                    proxy_model_loss.detach().cpu().numpy()
                    - irreducible_loss.detach().cpu().numpy(),
                    loss.detach().cpu().numpy()
                    - irreducible_loss.detach().cpu().numpy(),
                )[0, 1]
                self.log(
                    "cov_proxy_redloss_large_redloss",
                    cov,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )

                # Correlations between different metrics

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, loss
                )
                self.log(
                    "spearman_proxy_redloss_large_loss",
                    spearman,
                    on_step=True,
                    on_epoch=False,
                    logger=True,
                )

                spearman = self.spearman_correlation(proxy_model_loss, irreducible_loss)
                self.log(
                    "spearman_proxy_loss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                spearman = self.spearman_correlation(
                    proxy_model_loss - irreducible_loss, irreducible_loss
                )
                self.log(
                    "spearman_proxy_redloss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                spearman = self.spearman_correlation(loss, irreducible_loss)
                self.log(
                    "spearman_large_loss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                spearman = self.spearman_correlation(
                    loss - irreducible_loss, irreducible_loss
                )
                self.log(
                    "spearman_large_redloss_irrloss",
                    spearman,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                )

                # Standard deviations
                std = torch.std(loss - irreducible_loss)
                self.log(
                    "std_large_redloss", std, on_step=True, on_epoch=False, logger=True
                )

                std = torch.std(proxy_model_loss - irreducible_loss)
                self.log(
                    "std_proxy_redloss", std, on_step=True, on_epoch=False, logger=True
                )

    def on_train_epoch_end(self):
        
        if (self.scheduler_config is not None) and \
            (self.scheduler_step_on_step is False):
            
            scheduler = self.lr_schedulers()
            scheduler.step()
            
            self.log(
                'lr_rate', scheduler.get_lr()[0], on_step=False, on_epoch=True, logger=True
                )
        
        if self.track_diversity:
        
            print('logging index counts')
            
            df_data = pd.DataFrame(
                        {'selected_index_counts': pd.Series(self.selected_index_counts),
                         'target_index_counts': pd.Series(self.target_index_counts)})
                      
            selected_index_counts_table = wandb.Table(dataframe=df_data)
              
            #log with the underlying logger as the data format is non-standard:
            self.logger.experiment.log(
                {'selected_index_counts_table': selected_index_counts_table})
            
        #Log the per class training losses across the epoch:
        for c in range(self.num_groups):
            
            if self.training_points_per_label[c] > 0:
            
                #Calculate training accuracy and losses
                class_train_acc = self.training_acc_per_label[c] / self.training_points_per_label[c]
                class_train_loss = self.training_loss_per_label[c] / self.training_points_per_label[c]
                            
                #Log the worst class validation accuracy:
                self.log(f"class_{c}_train_loss",
                         class_train_loss,
                         on_epoch=True,
                         logger=True)
                
                self.log(f"class_{c}_train_acc",
                         class_train_acc,
                         on_epoch=True,
                         logger=True)
        
        #Reset the validation acc and points per label trackers...
        self.training_acc_per_label = np.zeros(self.num_groups)
        self.training_loss_per_label = np.zeros(self.num_groups)
        self.training_points_per_label = np.zeros(self.num_groups)
            
        
        
    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
              
        start_time = time.time()
        
        global_index, data, target, categories = process_batch(batch)
        
        # #For dictionary NLP data
        # if isinstance(batch, dict):
        #     global_index = batch.pop("idx")
        #     inputs = batch
        #     data = inputs
        #     target = inputs["labels"]
        #     categories = inputs.pop("categories")
        # else:        
        #     global_index, data, target, categories = batch
            
        if self.selection_method.bald:
            self.large_model.eval()
            enable_dropout(self.large_model)
            predictions = torch.zeros(
                (self.hparams.num_mc, len(data), 10), device=self.device
            )
            for i in range(self.hparams.num_mc):
                predictions[i] = self.large_model(data)
            predictions = predictions.transpose(0, 1)
            logits = torch.logsumexp(predictions, dim=1) - math.log(self.hparams.num_mc)
            loss = self.loss(logits, target)

        else:
                        
            logits = self.large_model(data)
            
            #Re-write the NLP code
            if isinstance(logits, SequenceClassifierOutput):
                logits = logits[1]            
            loss = self.loss(logits, target)
            
            
        if (isinstance(self.selection_method, robust_reducible_loss_selection)) or \
           (isinstance(self.selection_method, weighted_reducible_loss_selection)) or \
           (isinstance(self.selection_method, robust_payoff_reducible_loss_selection)):
            #Collect the validation losses for the robust rho loss selection
            #Entire Val dataset is eval'd after each training data step
            #See: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
            
            #For each class in the dataset count the number of instances and the losses:
            for c in range(self.num_groups):                
                self.num_points_in_class[c] += (categories == c).sum().cpu()
                self.running_sum_of_robust_holdout_losses[c] += loss[categories == c].sum().cpu() 
            

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, target, task="multiclass", num_classes=self.num_classes)
        self.log(
            "val_loss_epoch",
            loss.mean(),
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        
        #Calculate the worst class loss at this step
        
        #Record the per class validation loss
        for c in range(self.num_groups):
            
            #Setup assertion prevent nan's in this term
            self.log(f'class_{c}_val_loss',
                     loss[categories==c].mean(),
                     on_step=True,
                     on_epoch=True,
                     logger=True)
                      
            try: 
                class_acc = accuracy(preds[categories==c],
                                     target[categories==c], task="multiclass", num_classes=self.num_classes)
                
                self.log(f'class_{c}_val_acc',
                         class_acc,
                         on_step=True,
                         on_epoch=True,
                         logger=True)
                
                self.validation_acc_per_label[c] += ((categories==c).sum() * class_acc).\
                                                    detach().cpu().numpy()
                self.validation_points_per_label[c] += (categories==c).sum().\
                                                    detach().cpu().numpy()
                
            except RuntimeError as e:
                #RuntimeError thrown when len(preds[target==i]) == 0
                assert len(preds[categories==c]==0) or len(target[categories==c])==0,\
                    f"{e} thrown when neither preds or target have length zero" 
            
        if self.hparams.update_irreducible:
                        
            irred_loss_log_outputs = self.irreducible_loss_generator.\
                                            calculate_losses_and_accuracy(
                                                data=data,
                                                target=target,
                                                global_index=global_index, 
                                                category=categories)
                        
            for i, (irred_loss_mean, irred_loss_acc) in enumerate(irred_loss_log_outputs):
                
                self.log(
                    f"irlomo_val_loss_{i}",
                    irred_loss_mean,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
                self.log(
                    f"irlomo_val_acc_{i}",
                    irred_loss_acc,
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    prog_bar=True,
                )
            
        if self.proxy_model is not None:
            logits = self.proxy_model(data)
            proxy_loss = self.loss(logits, target)
            preds = torch.argmax(logits, dim=1)
            proxy_acc = accuracy(preds, target, task="multiclass", num_classes=self.num_classes)
            self.log(
                "proxy_val_loss_epoch",
                proxy_loss.mean(),
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            self.log(
                "proxy_val_acc_epoch",
                proxy_acc,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
            
        
        self.log('validation_step_time', time.time() - start_time,
                 on_step=True, on_epoch=True, logger=True)
        
        return loss.mean()
    
    def on_validation_epoch_end(self):
        
        #Robust holdout loss term calculation :
        if (isinstance(self.selection_method, robust_reducible_loss_selection)) or \
            (isinstance(self.selection_method, weighted_reducible_loss_selection))or \
            (isinstance(self.selection_method, robust_payoff_reducible_loss_selection)):
            
            assert not (self.num_points_in_class == 0).any(),\
                f"self.num_points_in_class are zero {self.num_points_in_class}"
            
            #Calculate the average holdout loss per class given the current model:
            if self.robust_holdout_loss_sum:
                self.robust_holdout_loss = self.running_sum_of_robust_holdout_losses
            else:                
                self.robust_holdout_loss = (self.running_sum_of_robust_holdout_losses\
                                        / self.num_points_in_class).cpu()    
            
                
            #Reset the running sum of losses and no. points:
            self.running_sum_of_robust_holdout_losses = torch.zeros(self.num_groups).cpu()
            self.num_points_in_class = torch.zeros(self.num_groups).cpu()
            
        #We should log the worst class validation loss here:
        val_acc_per_epoch = self.validation_acc_per_label/self.validation_points_per_label             
        
        #Log the worst class validation accuracy:
        self.log("worst_val_acc",
                 val_acc_per_epoch.min(),
                 on_epoch=True,
                 logger=True)
        
        #Reset the validation acc and points per label trackers...
        self.validation_acc_per_label = np.zeros(self.num_groups)
        self.validation_points_per_label = np.zeros(self.num_groups)
        
        if self.run_test_eval:
            self.running_test_eval()       
                
    
    def running_test_eval(self):
        
        #t = tensor.rand(2,2, device=self.device)
        
        dataloader = self.datamodule.test_dataloader()
        
        #Metrics to track during eval:
        running_loss_metric = 0
        running_accuracy_metric= 0
        points_per_class = np.zeros(self.num_classes)
        accuracy_per_class = np.zeros(self.num_classes)
        loss_per_class = np.zeros(self.num_classes)
        
        
        for batch in dataloader:
                       
            global_index, data, target, categories = process_batch(batch)
            
            # if isinstance(batch, dict):
            #     global_index = batch.pop("idx")
            #     inputs = batch
            #     data = inputs
            #     target = inputs["labels"]
            #     categories = inputs.pop("categories")
                
            # else:        
            #     global_index, data, target, categories = batch
                
            if isinstance(data, dict):
                for key in data.keys():
                    data[key] = data[key].to(self.device)
            else:
                data = data.to(self.device)
                
            global_index = global_index.to(self.device)
            #data         = data.to(self.device)
            target       = target.to(self.device)
            categories   = categories.to(self.device)
            
            #Calculate loss and target
            logits = self.large_model(data)
            loss = self.loss(logits, target)
            
            #metrics
            preds = torch.argmax(logits, dim=1)
            acc = accuracy(preds, target, task="multiclass", num_classes=self.num_classes)
            
            running_loss_metric += loss.sum()
            running_accuracy_metric += acc * len(target)
            
            for c in range(self.num_classes):
                
                if (target == c).any():
                    
                    points_per_class[c] += (target == c).sum()
                    
                    class_accuracy = accuracy(preds[target == c],
                                              target[target == c], task="multiclass", num_classes=self.num_classes)
                    
                    accuracy_per_class[c] += (class_accuracy * (target == c).sum())
                    
                    class_loss = self.loss(logits[target==c],
                                           target[target==c])
                    
                    loss_per_class[c] += class_loss.sum().detach().cpu().numpy()
                    
        #Log the metrics:                    
        self.log("test_eval_acc",
                 running_accuracy_metric/len(dataloader.dataset),
                 on_epoch=True,
                 logger=True)
            
        self.log("test_eval_loss",
                 running_loss_metric/len(dataloader.dataset),
                 on_epoch=True,
                 logger=True)
        
        loss_per_class /= points_per_class
        accuracy_per_class /= points_per_class
        
        for c in range(self.num_classes):
            
            self.log(f"test_eval_loss_class_{c}",
                     loss_per_class[c],
                     on_epoch=True,
                     logger=True)
            
            self.log(f"test_eval_acc_class_{c}",
                     accuracy_per_class[c],
                     on_epoch=True,
                     logger=True)
        
        

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        
        _, data, target, _ = process_batch(batch)
        
        # #For dictionary NLP data
        # if isinstance(batch, dict):
        #     batch.pop("idx")
        #     batch.pop("categories")
        #     inputs = batch
        #     data = inputs
        #     target = inputs["labels"]
            
        
        # else:
        
        #     _, data, target, _ = batch
        
        
        logits = self.large_model(data)
        
        if isinstance(logits, SequenceClassifierOutput):
            logits = logits[1]
        
        loss = self.loss(logits, target).mean()

        # validation metrics
        preds = torch.argmax(F.log_softmax(logits, dim=1), dim=1)
        acc = accuracy(preds, target, task="multiclass", num_classes=self.num_classes)
        
        #Add per group loss logging such that we can solely use Wandb as a storage for model results...
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss.mean()

    def configure_optimizers(self):
        
        if self.proxy_model is not None:
            opt_large_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.large_model.parameters(),
                _convert_="partial",
            )
            opt_proxy_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.proxy_model.parameters(),
                _convert_="partial",
            )
            return [opt_large_model, opt_proxy_model]
        
        if self.hparams.update_irreducible:
            
            opt_large_model = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.large_model.parameters(),
                _convert_="partial",
            )
            
            # config optimizers of irreducible loss generator:                   
            self.irreducible_loss_generator.config_optimizers(
                optim_config=self.optimizer_config,
                loss=self.loss)                                
        
            return opt_large_model #might need to make this a list []
        else:
            optimizer = hydra.utils.instantiate(
                config=unmask_config(self.optimizer_config),
                params=self.large_model.parameters(),
                _convert_="partial",
            )
            
            if self.scheduler_config is not None:
                
                #If total iters flag exists set it to the number of training steps:
                if self.scheduler_config.get('total_iters', None) is not None:
                    
                    print(self.scheduler_config)
                    print(self.scheduler_config['total_iters'])
                    self.scheduler_config['total_iters'] = self.number_training_steps                    
                    print(self.scheduler_config['total_iters'])
            
                lr_scheduler = hydra.utils.instantiate(
                    config=unmask_config(self.scheduler_config),
                    optimizer=optimizer,
                    )
            
                return ([optimizer], [lr_scheduler])
            
            else:
                
                return ([optimizer])
        
    def on_save_checkpoint(self, checkpoint):
                
        #Save the model checkpoint using the model_io
        self.model_io.save_checkpoint(checkpoint, irreducible_model=False)

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp, device=x.device)
        ranks[tmp] = torch.arange(len(x), device=x.device)
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor):
        """Compute correlation between 2 1-D vectors
        Args:
            x: Shape (N, )
            y: Shape (N, )
        """
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        x_rank = self._get_ranks(x)
        y_rank = self._get_ranks(y)

        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        return 1.0 - (upper / down)

    def log_repetition(self, data):
        """Measure repetition of selected points in previous epochs. Given current indices selected, logs what
        percentage of them were also selected exactly 1, 5, and 20 epochs ago. Requires shuffle=False."""
        assert self.datamodule.hparams.shuffle == False
        selected_batch_size = int(len(data))
        train_set_size = self.datamodule.indices_train.sequence.size
        epoch_size = int(train_set_size * self.hparams.percent_train)
        selected_indices_now = self.sequence[-selected_batch_size:]

        selected_indices_1_epoch_ago = self.sequence[
            -1 * epoch_size - selected_batch_size : -1 * epoch_size
        ]
        selected_indices_5_epoch_ago = self.sequence[
            -5 * epoch_size - selected_batch_size : -5 * epoch_size
        ]
        selected_indices_20_epoch_ago = self.sequence[
            -20 * epoch_size - selected_batch_size : -20 * epoch_size
        ]

        perct_repeated_1_epoch_ago = np.intersect1d(
            selected_indices_now, selected_indices_1_epoch_ago
        ).size / float(selected_batch_size)
        perct_repeated_5_epoch_ago = np.intersect1d(
            selected_indices_now, selected_indices_5_epoch_ago
        ).size / float(selected_batch_size)
        perct_repeated_20_epoch_ago = np.intersect1d(
            selected_indices_now, selected_indices_20_epoch_ago
        ).size / float(selected_batch_size)

        self.log(
            "perct_idx_repeated_from_1_epoch_ago",
            perct_repeated_1_epoch_ago,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "perct_idx_repeated_from_5_epoch_ago",
            perct_repeated_5_epoch_ago,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            "perct_idx_repeated_from_20_epoch_ago",
            perct_repeated_20_epoch_ago,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
