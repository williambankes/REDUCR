from typing import List, Optional

import wandb
import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import DummyLogger

from src.utils import utils
from src.models.IrreducibleLossGenerators import (
    IrreducibleLossGeneratorFactory,
    PrecomputedIrreducibleLossGenerator)  

from src.datamodules.datamodules import CIFAR100DataModule  

from src.utils.io_utils import IOFactory

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init IOFactory:
    log.info('Instantiating IOFactory')    
    iofactory: IOFactory = hydra.utils.instantiate(config.model_io,
                    selection_method=utils.process_config_for_target(config.selection_method),
                    datamodule_name=config.datamodule.get('datamodule_name',
                        utils.process_config_for_target(config.datamodule)),
                    imbalanced_class=config.sampler_factory.get('imbalanced_class', None), #TO DO - change to imbalanced_class
                    eta=config.selection_method.get('eta', None),
                    beta=config.selection_method.get('beta', None),
                    multi_headed_model=config.irreducible_loss_generator.get('multi_headed_model', False))
    model_io = iofactory.create_model_io()
    
    # init irreducible loss generator factory:
    irreducible_loss_generator_factory: IrreducibleLossGeneratorFactory =\
        hydra.utils.instantiate(config.irreducible_loss_generator,
                                selection_method=utils.process_config_for_target(config.selection_method),
                                weights=config.selection_method.get('weights', None),
                                num_categories=config.selection_method.get('num_categories', None),
                                model_io=model_io)
    irreducible_loss_generator = irreducible_loss_generator_factory.create_loss_generator()

    # If irreducible losses are precomputed check the precomputed input:            
    if isinstance(irreducible_loss_generator, PrecomputedIrreducibleLossGenerator):
        
        #add nlp_field to datamodule:
        with open_dict(config):
            config.datamodule.model_name_or_path=config.model.\
                large_model.get('model_name_or_path', None)

        #Check the precomputed irred losses:
        irreducible_loss_generator.check_precomputed_irreducible_losses(config.datamodule)
    
    # Set seed again, so that the main datamodule is instantiated with the
    # same random seed whether or not the precomputed losses are used
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    #print('Datamodule config:' config.model.large_model)
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule,
                                                  model_name_or_path=config.model.\
                                                  large_model.get('model_name_or_path', 
                                                                  None))
    
    # Init sampler_factory
    log.info("Instantiating sampler_factory")
    sampler_factory = hydra.utils.instantiate(config.sampler_factory,
                                              num_classes=datamodule.num_classes)
    log.info("Setup datamodule")
    datamodule.setup(sampler_factory=sampler_factory,
                     stage=config.datamodule.get('stage', None),
                     test=config.get('test', False))
    
    #Calculate the number training steps:
    number_training_steps = (len(datamodule.train_dataloader().dataset)/config.datamodule.batch_size)\
        * config.trainer.max_epochs

    # init selection method
    log.info(f"Instantiating selection method <{config.selection_method._target_}>")
    selection_method = hydra.utils.instantiate(config.selection_method)

    # Init lightning model
    log.info("Instantiating models")
    pl_model: LightningModule = hydra.utils.instantiate(
        config.model,
        selection_method=selection_method,
        irreducible_loss_generator=irreducible_loss_generator,
        datamodule=datamodule,
        update_irreducible=config.irreducible_loss_generator.get('update_irreducible', False),
        num_classes= datamodule._get_set_of_labels(), #num_classes
        num_groups =20 if isinstance(datamodule, CIFAR100DataModule) else None, #num_groups
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        scheduler_config=utils.mask_config(
            config.get("scheduler", None)),
        number_training_steps = number_training_steps,
        model_io=model_io,
        _convert_="partial",
        large_model={
            'task_name' : config.datamodule.get('task_name', None),
            'num_labels': datamodule.num_classes},
        seed=config.seed
    )

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[DummyLogger] = []
    if "logger" in config:
        for key, lg_conf in config.logger.items():              
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
            if key == "wandb_key":
                wandb.login(key=lg_conf)
                
    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send config to all lightning loggers
    log.info("Logging hyperparameters!")
    trainer.logger.log_hyperparams(config)

    if config.eval_set == "val":
        val_dataloader = datamodule.val_dataloader()
    elif config.eval_set == "test":
        val_dataloader = datamodule.test_dataloader()
        log.warning(
            "Using the test set as the validation dataloader. This is for final figures in the paper"
        )
        
    # Continue training setup
    if config.get('continue', None):
        ckpt_path = model_io.create_load_checkpoint_path(irreducible_model=False)
    else:
        ckpt_path = None

    # Train the model
    log.info("Starting training!")
    log.info("\nTraining on main branch\n")
    print("\nTraining on main branch\n")
    trainer.fit(
        pl_model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path
    )

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(dataloaders=datamodule.test_dataloader())
        
    log.info("Evaluating and saving model from checkpoint path")
    model = model_io.load_checkpoint(irreducible_model=False)
    model.eval()
    model.cpu() #Move the model to the cpu for eval as the trainer isn't used to map the dataset onto the gpu
    
    #We turn off the sample flag to ensure each datapoint is seen only once:
    target_model_test_and_checks = utils.compute_losses_with_sanity_checks(
                                    dataloader=datamodule.test_dataloader(), model=model)
    
    model_io.save_losses_and_checks(target_model_test_and_checks,
                                    irreducible_model=False)
    
    log.info(f"Using monitor: {trainer.checkpoint_callback.monitor}")
    
    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=pl_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
