import wandb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import DummyLogger

from src.utils import utils
from src.utils.io_utils import IOFactory

from src.datamodules.datamodules import CIFAR100DataModule

from typing import List, Optional

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Init IOFactory:
    log.info('Instantiating IOFactory')
    iofactory: IOFactory = hydra.utils.instantiate(config.model_io,
                    datamodule_name=config.datamodule.get('datamodule_name',
                        utils.process_config_for_target(config.datamodule)),
                    imbalanced_class=config.sampler_factory.get('imbalanced_class', None),
                    gradient_weighted_class=config.model.get('gradient_weighted_class', None),
                    multi_headed_model=config.model.get('multi_headed_model', False))
    
    model_io = iofactory.create_model_io()


    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule,
                                                              model_name_or_path=config.model.\
                                                              model.get('model_name_or_path', None))
    
    log.info("Instantiating sampler_factory")
    sampler_factory = hydra.utils.instantiate(config.sampler_factory,
                                              num_classes=datamodule.num_classes)

    log.info("Setup datamodule")
    datamodule.setup(sampler_factory=sampler_factory,
                     stage=config.datamodule.get('stage', None),
                     test=config.get('test', False))
    
    #Calculate the number training steps:
    number_training_steps = (len(datamodule.val_dataloader().dataset)/config.datamodule.batch_size)\
        * config.trainer.max_epochs

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    pl_model: LightningModule = hydra.utils.instantiate(
        config=config.model,
        optimizer_config=utils.mask_config(
            config.get("optimizer", None)
        ),  # When initialising the optimiser, you need to pass it the model parameters. As we haven't initialised the model yet, we cannot initialise the optimizer here. Thus, we need to pass-through the optimizer-config, to initialise it later. However, hydra.utils.instantiate will instatiate everything that looks like a config (if _recursive_==True, which is required here bc OneModel expects a model argument). Thus, we "mask" the optimizer config from hydra, by modifying the dict so that hydra no longer recognises it as a config.
        scheduler_config=utils.mask_config(
            config.get("scheduler", None)
        ),  # see line above
        number_training_steps = number_training_steps,
        model_io=model_io,
        datamodule=datamodule,
        number_of_classes=datamodule._get_set_of_labels(),
        number_of_groups=20 if isinstance(datamodule, CIFAR100DataModule) else None,
        _convert_="partial",
        model={
            'task_name' : config.datamodule.get('task_name', None),
            'num_labels': datamodule.num_classes},
        loaded_from_checkpoint=False
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

    # Train the model
    log.info("Starting training!")
    trainer.fit(
        pl_model,
        train_dataloaders=datamodule.val_dataloader(),
        val_dataloaders=datamodule.train_dataloader(),#changed to train dataloader
        #Should this be the test dataloader particularly as we don't allow sampling on the test set?
    )

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(dataloaders=datamodule.test_dataloader())

    ## Use this if you want to compute the irred losses for a model you have already trained
    # trainer.checkpoint_callback.best_model_path = "/path/to/model"
       
    log.info("Evaluating and saving model from checkpoint path")
    model = model_io.load_checkpoint(irreducible_model=True)
    model.eval()
    model.cpu() #move the model to the cpu for eval on the test set...
    
    #We turn off the sample flag to ensure each datapoint is seen only once:
    if config.model.get('multi_headed_model', False):
                
        irreducible_loss_and_checks = utils.compute_losses_with_sanity_checks_multi_head(
                dataloader=datamodule.train_dataloader(sample=False, keep_test_on=True),
                model=model, num_classes=datamodule.num_classes)
        
        model_io.save_losses_and_checks(irreducible_loss_and_checks,
                                        irreducible_model=True)
    else:
        irreducible_loss_and_checks = utils.compute_losses_with_sanity_checks(
            dataloader=datamodule.train_dataloader(sample=False), model=model)
        
        model_io.save_losses_and_checks(irreducible_loss_and_checks,
                                        irreducible_model=True)        
    
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

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
