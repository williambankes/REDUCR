import logging
import os
import warnings
from typing import List, Sequence, Union, Tuple
import subprocess
from transformers.modeling_outputs import SequenceClassifierOutput

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn as nn
import numpy as np
import tqdm
import gcsfs

from src.datamodules.clothing1m import Clothing1M

def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def process_batch(batch: Union[dict, Tuple]) -> Tuple:
    
    if isinstance(batch, dict):
        global_index = batch.pop("idx")
        inputs = batch
        data = inputs
        target = inputs["labels"]
        categories = target
        
    else:
        global_index, data, target, categories = batch
        
    return global_index, data, target, categories
        

def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "selection_method",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "optimizer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.Logger],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """
    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    if "selection_method" in config:
        hparams["selection_method"] = config["selection_method"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.logger.DummyLogger],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


### -------------------------------------------------------------------
# utils for temporarily masking a config dict so that hydra won't automatically
# instatiate it and instead passes it through for later instatiation.


def mask_config(config):
    
    print('config:', config)
    print('config type', type(config))
    
    """Mask config from hydra instantiation function by removing "_target_" key."""
    if config is not None:
        with open_dict(
            config
        ):  # this is needed to be able to edit omegaconf structured config dicts
            if "_target_" in config.keys():
                config["target"] = config.pop("_target_")
    return config


def unmask_config(config):
    """Re-introduce "_target_" key so that hydra instantiation function can be used"""
    if config is not None:
        if "target" in config.keys():
            config["_target_"] = config.pop("target")
    return config


### -------------------------------------------------------------------
### utils for precomputing the irreducible loss just once
### and then reusing it across all main model runs


def compute_losses_with_sanity_checks(dataloader, model, device=None, multi_headed_model_head:Union[int, bool]=False):
    """Compute losses for full dataset.

    (I did not implement this with
    trainer.predict() because the trainset returns (index, x, y) and does not
    readily fit into the forward method of the irreducible loss model.)

    Returns:
        losses: Tensor, losses for the full dataset, sorted by <globa_index> (as
        returned from the train data loader). losses[global_index] is the
        irreducible loss of the datapoint with that global index. losses[idx] is
        nan if idx is not part of the dataset.
        targets: Tensor, targets of the datsets, sorted by <index> (as
        returned from the train data loader). Also see above.This is just used to verify that
        the data points are indexed in the same order as they were in this
        function call, when <losses> is used.
    """
    if isinstance(dataloader.dataset, Clothing1M):
        if device is None:
            device = model.device
        else:
            model.to(device)
        print("Computing irreducible loss full training dataset.")
        idx_of_control_images = [1, 3, 10, 30, 100, 300, 1000, 3000]
        control_images = [0] * len(idx_of_control_images)
        losses = torch.zeros(len(dataloader.dataset)).type(torch.FloatTensor)
        targets = torch.zeros(len(dataloader.dataset)).type(torch.LongTensor)
        prediction_correct = []
        preds = torch.zeros(len(dataloader.dataset)).type(torch.LongTensor)
        logits_list = torch.zeros((len(dataloader.dataset), 14)).type(torch.FloatTensor)
        
        with torch.inference_mode():
            for idx, x, target, _ in tqdm.tqdm(dataloader):
                idx, x, target = idx.long(), x.to(device), target.to(device)
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, target, reduction="none")
                losses[idx] = loss.cpu()
                targets[idx] = target.cpu()
                prediction_correct.append(torch.eq(torch.argmax(logits, dim=1), target).cpu())
                preds[idx] = torch.argmax(logits, dim=1)
                logits_list[idx] = logits

                for (id, image) in zip(idx, x):
                    if id in idx_of_control_images:
                        local_index = idx_of_control_images.index(id)
                        control_images[local_index] = image.cpu()

        acc = torch.cat(prediction_correct, dim=0)
        acc = acc.type(torch.FloatTensor).mean()
        average_loss = losses.mean()

        log.info(
            f"Accuracy of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {acc:.3f}\n"
            f"Average loss of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {average_loss:.3f}"
        )
        
    
        output = {
            "irreducible_losses": losses,
            "sorted_targets": targets,
            "idx_of_control_images": idx_of_control_images,
            "control_images": control_images,
            "heldout_accuracy": acc,
            "heldout_average_loss": average_loss,
            "preds": preds,
            "logits": logits 
        }

        return output
    else:
        print("Computing irreducible loss full training dataset.")
        idx_of_control_images = [1, 3, 10, 30, 100, 300, 1000, 3000]
        control_images = [0] * len(idx_of_control_images)
        losses = []
        idxs = []
        targets = []
        categories = []
        preds = []
        prediction_correct = []
        logits_list = []

        with torch.inference_mode():
            for batch in dataloader:
                
                idx, x, target, category = process_batch(batch)
                
                logits = model(x)
                
                if isinstance(logits, SequenceClassifierOutput):
                    logits = logits[1]
                                
                loss = nn.functional.cross_entropy(logits, target, reduction="none")
                logits_list.append(logits)
                losses.append(loss)
                idxs.append(idx)
                targets.append(target)
                categories.append(category)
                preds.append(torch.argmax(logits, dim=1))
                prediction_correct.append(torch.eq(torch.argmax(logits, dim=1), target))

                for (id, image) in zip(idx, x):
                    if id in idx_of_control_images:
                        local_index = idx_of_control_images.index(id)
                        control_images[local_index] = image

        num_classes = logits.shape[1]
        acc = torch.cat(prediction_correct, dim=0)
        acc = acc.type(torch.FloatTensor).mean()
        average_loss = torch.cat(losses, dim=0).type(torch.FloatTensor).mean()

        log.info(
            f"Accuracy of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {acc:.3f}\n"
            f"Average loss of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {average_loss:.3f}"
        )

        logits_list_temp = torch.cat(logits_list, dim=0)
        losses_temp = torch.cat(losses, dim=0)
        idxs = torch.cat(idxs, dim=0)
        targets_temp = torch.cat(targets, dim=0)
        preds_temp = torch.cat(preds, dim=0)
        categories_temp = torch.cat(categories, dim=0)

        max_idx = idxs.max()

        losses = torch.tensor(
            [float("nan")] * (max_idx + 1), dtype=losses_temp.dtype
        )  # losses[global_index] is the irreducible loss of the datapoint with that global index. losses[idx] is nan if idx is not part of the dataset.
        targets = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
        preds = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
        categories = torch.zeros(max_idx + 1, dtype=categories_temp.dtype)
        logits_list = torch.zeros((max_idx + 1, num_classes), dtype=logits_list_temp.dtype)
        
        losses[idxs] = losses_temp
        targets[idxs] = targets_temp
        preds[idxs] = preds_temp
        categories[idxs] = categories_temp
        logits_list[idxs,:] = logits_list_temp

        output = {
            "irreducible_losses": losses,
            "sorted_targets": targets,
            "preds":preds,
            "categories":categories,
            "idx_of_control_images": idx_of_control_images,
            "control_images": control_images,
            "heldout_accuracy": acc,
            "heldout_average_loss": average_loss,
            "logits":logits_list
        }

        return output
    
    
    
def compute_losses_with_sanity_checks_multi_head(dataloader, model, num_classes:int):
    """Compute losses for full dataset.

    (I did not implement this with
    trainer.predict() because the trainset returns (index, x, y) and does not
    readily fit into the forward method of the irreducible loss model.)

    Returns:
        losses: Tensor, losses for the full dataset, sorted by <globa_index> (as
        returned from the train data loader). losses[global_index] is the
        irreducible loss of the datapoint with that global index. losses[idx] is
        nan if idx is not part of the dataset.
        targets: Tensor, targets of the datsets, sorted by <index> (as
        returned from the train data loader). Also see above.This is just used to verify that
        the data points are indexed in the same order as they were in this
        function call, when <losses> is used.
    """
        
    print("Computing irreducible loss full training dataset.")
    idx_of_control_images = [1, 3, 10, 30, 100, 300, 1000, 3000]
    control_images = [0] * len(idx_of_control_images)
    
    #standard metric lists:
    idxs = []
    targets = []
    categories = []
    
    #init multiple metric lists:
    losses = [ list() for _ in range(num_classes)] 
    preds = [ list() for _ in range(num_classes)]
    prediction_correct = [ list() for _ in range(num_classes)]
    logits_list = [ list() for _ in range(num_classes)]

    with torch.inference_mode():
        for batch in dataloader:
            
            idx, x, target, category = process_batch(batch)
            
            logits = model(x)
                        
            # if isinstance(logits, SequenceClassifierOutput):
            #     logits = logits[1]
                
            #For each of the classes calculate the class specific metric:
            for c in range(num_classes):
                loss = nn.functional.cross_entropy(logits[c], target, reduction="none")
                logits_list[c].append(logits[c])
                losses[c].append(loss)
                preds[c].append(torch.argmax(logits[c], dim=1))
                prediction_correct[c].append(torch.eq(torch.argmax(logits[c], dim=1), target))
                
            idxs.append(idx)
            targets.append(target)
            categories.append(category)

            for (id, image) in zip(idx, x):
                if id in idx_of_control_images:
                    local_index = idx_of_control_images.index(id)
                    control_images[local_index] = image
    
    outputs = list()
    
    #Setup the idx tensor outside the for loop:
    idxs = torch.cat(idxs, dim=0)
    
    for c in range(num_classes):
    
        acc = torch.cat(prediction_correct[c], dim=0)
        acc = acc.type(torch.FloatTensor).mean()
        average_loss = torch.cat(losses[c], dim=0).type(torch.FloatTensor).mean()
    
        log.info(
            f"Accuracy of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {acc:.3f}\n"
            f"Average loss of irreducible loss model on train set (i.e. the train set of the target model, not the train set of the irreducible loss model) is {average_loss:.3f}"
        )
        
        #Combine lists into torch tensors:
        logits_list_temp = torch.cat(logits_list[c], dim=0)
        losses_temp = torch.cat(losses[c], dim=0)
        targets_temp = torch.cat(targets, dim=0)
        preds_temp = torch.cat(preds[c], dim=0)
        categories_temp = torch.cat(categories, dim=0)
        max_idx = idxs.max()
        
        #Process tensors:
        losses_out = torch.tensor(
            [float("nan")] * (max_idx + 1), dtype=losses_temp.dtype
        )  # losses[global_index] is the irreducible loss of the datapoint with that global index. losses[idx] is nan if idx is not part of the dataset.
        targets_out = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
        preds_out = torch.zeros(max_idx + 1, dtype=targets_temp.dtype)
        categories_out = torch.zeros(max_idx + 1, dtype=categories_temp.dtype)
        logits_list_out = torch.zeros((max_idx + 1, num_classes), dtype=logits_list_temp.dtype)
        
        
        losses_out[idxs] = losses_temp
        targets_out[idxs] = targets_temp
        preds_out[idxs] = preds_temp
        categories_out[idxs] = categories_temp
        logits_list_out[idxs,:] = logits_list_temp
    
        output = {
            "irreducible_losses": losses_out,
            "sorted_targets": targets_out,
            "preds":preds_out,
            "categories":categories_out,
            "idx_of_control_images": idx_of_control_images,
            "control_images": control_images,
            "heldout_accuracy": acc,
            "heldout_average_loss": average_loss,
            "logits":logits_list_out
        }
        outputs.append(output)

    return outputs

def verify_correct_dataset_order(
    dataloader,
    sorted_target,
    idx_of_control_images,
    control_images,
    dont_compare_control_images=False,
):
    """Roughly checks that a dataloader is sorted in the same order as the
    precomputed losses. Concretely, does two checks: 1) that the labels used for
    computing the irreducible losses are in the same order as those returned by
    the dataloader. 2) That a handful of example images is identical across the
    ones used for computing the irreducble loss and the ones returned by the
    dataloader.

    Args:
        dataloader: a PyTorch dataloader, usually the training dataloader in our
        current setting.
        sorted_target, idx_of_control_images, control_images: those were saved
        as controls when pre-computing the irreducible loss.
        dont_compare_control_images: bool. Set to True if you don't want to compare
        control images (required if there is trainset augmentation)
    """
    print(
        "Verifying that the dataset order is compatible with the order of the precomputed losses."
    )
    if isinstance(dataloader.dataset, Clothing1M):
        for idx in np.random.choice(range(len(dataloader.dataset)), 10000, replace=False):
            _, _, target, _ = dataloader.dataset[idx]

            #change target to type tensor for torch.equal comparison:            
            target = torch.tensor(target)
            
            assert torch.equal(target, sorted_target[idx]), "Unequal Images. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."
            
        if not dont_compare_control_images:
            for i, idx in enumerate(idx_of_control_images):
                _, image, _ = dataloader.dataset[idx]
                assert torch.equal(image, control_images[i]), "Unequal Images. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."
    else:
        for batch in dataloader:
            
            idx, x, target, category = process_batch(batch)
            
            assert torch.equal(
                target, sorted_target[idx]
            ), f"Unequal labels. At {idx} target:{target} != {sorted_target[idx]}. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."
            if not dont_compare_control_images:
                for id, image in zip(idx, x):
                    if id in idx_of_control_images:
                        assert torch.equal(
                            image, control_images[idx_of_control_images.index(id)]
                        ), "Unequal Images. Order of dataloader is not consistent with order used when precomputing irreducible losses. Can't use precomputed losses. Either ask Jan, or use the irreducible loss model directly ('irreducible_loss-generator: irreducible_loss_model.yaml' in the config.). Note that the latter is probably slower."


def process_config_for_target(config):
    
    target_str = config['_target_'].split('.')[-1]
    
    return target_str


def load_gc_torch_files(project, path):
    
    fs = gcsfs.GCSFileSystem(project=project)
    
    with fs.open(path, 'rb') as f:
        return torch.load(f)    


def save_repo_status(path):
    """Save current commit hash and uncommitted changes to output dir."""

    with open(os.path.join(path, "git_commit.txt"), "w+") as f:
        subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)

    with open(os.path.join(path, "git_commit.txt"), "r") as f:
        commit = f.readline()

    with open(os.path.join(path, "workspace_changes.diff"), "w+") as f:
        subprocess.run(["git", "diff"], stdout=f)
    
    return commit

def create_irreducible_losses_path(home_dir, scratch_path, imbalanced, number,
                                   datamodule_name, gcp=False):
    
    if gcp:
        return gcp + '/' + datamodule_name + '/' + 'irreducible_models'
    else:
        if scratch_path != 'tutorial_outputs':
            #Find file where specific irred model file paths are saved:
            irred_model_file_name = "irred_model_path{0}_{1}.txt".format(
                '_im' if imbalanced else '',
                        number)
            irred_model_file_path = os.path.join(home_dir, irred_model_file_name)
            with open(irred_model_file_path, "r") as my_file:
                return os.path.dirname(my_file.readline())
        
        else:
            path = os.path.join(home_dir, 'tutorial_outputs', 
                                datamodule_name[:-10], 'irreducible_loss_model')
            return path
    

def create_irreducible_losses_config(local_path, imbalanced, number, 
                                      selection_method, num_classes=-1,
                                      gcp_project=False, single_class_weighted=None,
                                      update_irreducible=False):
    """
    Create file_path names to load irreducible loss generator models with

    Parameters
    ----------
    local_path : r'string
        path to the local directory containing the irreducible loss generator models
    imbalanced : bool
        bool if the irreducible loss model was trained on an imbalanced dataset
    number : int
        For parallel experiments number denotes different experiments save files.
        Also used when generated class_imbalanced datasets in paralle
    num_classes : int, optional
        Number of classes for the robust selection method to create files for. The default is -1.
    gcp_project : bool | str
        False if not using the gcp or a string detailing the gcp project if gcp being used
    single_class_weighted : None | int
        The single classes weighted if weighted reducible holdout loss used
    update_irreducible : bool
        If the irreducible loss model is a model with updated weights (True) 
        or a precomputed tensor (False)
    

    Raises
    ------
    NotImplementedError
        Imbalanced and robust selection are currently un implemented as both
        imbalance the irreducible loss model in different ways.

    Returns
    -------
    file_names: r'string or <list>r'string
        If robust selection not used returns a file path to load the equvialent 
        irreducible loss model.

    """
    
    if selection_method in ['robust_reducible_loss_selection', 
                            'weighted_reducible_loss_selection'] and imbalanced:
        raise NotImplementedError()
    
    
    if selection_method == 'uniform_selection':
        return None
       
    #Write all cases:
    if update_irreducible:
        
        if selection_method == 'weighted_reducible_loss_selection' and \
            (single_class_weighted is not None):
                
                file_names = ["irred_model_checkpoint" + \
                              f'_{number}_class_{single_class_weighted}.ckpt']
                list_order = [0]
        
        elif selection_method in ['robust_reducible_loss_selection', 
                                'weighted_reducible_loss_selection'] and \
            (single_class_weighted is None):
                
                raise NotImplementedError()
        
        else:
            
            file_names = [f"irred_model_checkpoint_{number}_.ckpt"]
            list_order = [0]
            
        #create update irreducible dictionary object:
        return [{"_target_": 'src.models.OneModel.OneModel.load_from_checkpoint',
                 "checkpoint_path": os.path.join(local_path, name)} for name in file_names],\
                list_order
            
            
    else: #Load irreducible model results from prior computation:
        
        if selection_method == 'weighted_reducible_loss_selection' and \
            (single_class_weighted is not None):
                
            file_names = ["irred_losses_and_checks" +\
                                      f'_{number}_class_{single_class_weighted}.pt']
            list_order = [0]
            
        elif selection_method in ['robust_reducible_loss_selection', 
                                'weighted_reducible_loss_selection'] and \
            (single_class_weighted is None):
            
            file_names = ["irred_losses_and_checks" + f'_{number}_class_{n}.pt'
                           for n in range(num_classes)]  
            list_order = [n for n in range(num_classes)]
        
        
        else:
            
            file_names = ["irred_losses_and_checks{0}_{1}.pt".format(
            '_im' if imbalanced else '', number)]
            list_order = [0]
        
        #Create actual dictionary objects here:
        if gcp_project:
            return [{"_target_": "src.utils.utils.load_gc_torch_files",
                     "project":gcp_project,
                     "path":os.path.join(local_path, name)} for name in file_names],\
                list_order

        else:
            return [{"_target_": "torch.load",
                     "f": os.path.join(local_path, name)} for name in file_names],\
                list_order
    
        
    
    
def process_irreducible_loss_generator(irreducible_loss_generators,
                                       selection_method,
                                       num_classes=None,
                                       single_class_weighted=None,
                                       update_irreducible=False):
    """
    Formats the irreducible loss generator correctly depending upon the 
    selection method used. In the case of weighted_reducible_loss_selection,
    when the single class weighted is not zero we only weight one class and thus only
    require one set of irreducible losses.

    Parameters
    ----------
    irreducible_loss_generators : list<torch.tensor>
        A list of torch tensors of the irreducible loss setting
    selection_method : str
        Selection methodology used in experiment
    single_class_weighted : int, optional
        The index of the single weighted class in the weighted_rho setting.
        The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    
    #run assertions on the input:
    if single_class_weighted is not None:
        assert num_classes is not None,\
            'Num classes must be specified when single_class_weighted is not None'
    
    
    # Two cases:
    if update_irreducible:
        
        if selection_method == 'robust_reducible_loss_selection':
            raise NotImplementedError()
            
        elif selection_method == 'weighted_reducible_loss_selection' and\
            single_class_weighted is None:
            
            raise NotImplementedError()
        
        elif selection_method == 'weighted_reducible_loss_selection' and\
            single_class_weighted is not None:
            
            raise NotImplementedError()
            
        else:
            return irreducible_loss_generators
    
    else:
        
        #Get the irreducible losses from the irred losses generators:
        irreducible_loss_generators = [l['irreducible_losses']for l in irreducible_loss_generators]
        
        if selection_method == 'robust_reducible_loss_selection':
            return irreducible_loss_generators
        
        elif selection_method == 'weighted_reducible_loss_selection' and\
            single_class_weighted is None:
            
                return irreducible_loss_generators
        
        elif selection_method == 'weighted_reducible_loss_selection' and\
            single_class_weighted is not None:
            
            zeros_irred_losses = torch.zeros(irreducible_loss_generators[0].shape)
            output = [zeros_irred_losses]*num_classes
            output[single_class_weighted] = irreducible_loss_generators[0] #only one element returned
            
            return output
        
        else:
            
            return irreducible_loss_generators[0]
        
        
        