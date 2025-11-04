#!/usr/bin/env python
from socket import gethostname
from getpass import getuser
import sys
from logging import getLogger, WARNING
from pathlib import Path
from aim.pytorch_lightning import AimLogger
import hydra
from hydra.utils import instantiate
import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning import seed_everything
from omegaconf import DictConfig
from omegaconf import OmegaConf
from coolname import generate_slug
from deepmuonreco.model import Model
from deepmuonreco.nn.utils import init_params


_logger = getLogger(__name__)

# FIXME: config?
getLogger('matplotlib').setLevel(WARNING)
getLogger('PIL').setLevel(WARNING)
getLogger('aim').setLevel(WARNING)
getLogger('filelock').setLevel(WARNING)



OmegaConf.register_new_resolver(
    'slug',
    lambda pattern = 2: generate_slug(pattern=pattern),
    use_cache=True,
    replace=True,
)

OmegaConf.register_new_resolver(
    name='eval',
    resolver=eval,
)

@hydra.main(
    config_path='./config',
    config_name='tracker_track_selection',
    version_base=None,
)
def main(config: DictConfig):
    _logger.info(' '.join(sys.argv))
    _logger.info(f'Host: {gethostname()}')
    _logger.info(f'User: {getuser()}')
    _logger.info(f'CWD: {Path.cwd()}')

    output_dir = Path(config.paths.run_dir)
    with open(output_dir / 'config.yaml', 'w') as stream:
        OmegaConf.save(config=config, f=stream)

    torch.set_num_threads(config.torch.num_threads)
    torch.set_num_interop_threads(config.torch.num_interop_threads)
    torch.set_float32_matmul_precision(precision=config.torch.float32_matmul_precision)

    seed_everything(seed=config.run.seed, workers=True)

    model = Model.from_config(config=config)
    _logger.debug(f'{model=}')
    _logger.info(f'{model.model=}')
    _logger.info(f'{model.num_params=}')

    _logger.info('Initializing model weights')
    model.model.apply(init_params)

    callback_dict = instantiate(config.callbacks)
    logger = instantiate(config.logger)

    trainer: Trainer = instantiate(config.trainer)(
        callbacks=list(callback_dict.values()),
        logger=logger,
    )
    datamodule: LightningDataModule = instantiate(config.datamodule)

    if isinstance(logger, AimLogger):
        logger.experiment.name = config.run.name

        logger.experiment.set(
            key='config',
            val=OmegaConf.to_container(config), # type: ignore
        )
        logger.experiment.set(
            key='env',
            val={
                'host': gethostname(),
                'cwd': str(Path.cwd()),
                'user': getuser(),
            },
        )
        logger.experiment.set(
            key='model',
            val={
                'num_params': model.num_params,
            }
        )
        for tag in config.run.tags:
            logger.experiment.add_tag(tag)

        description_file = output_dir / 'description.txt'
        if description_file.exists():
            _logger.info(f'Loading description from {description_file}')
            with open(description_file, 'r') as stream:
                description = stream.read()
            _logger.info(f'{description=}')
            logger.experiment.description = description
        elif description := config.run.description:
            logger.experiment.description = description

    if config.run.pre_fit_validation:
        # `validate` phase is supposed be run with the partial or full validation dataset
        trainer.validate(model=model, datamodule=datamodule)
    else:
        _logger.info(f'Skipping pre-fit validation as per {config.run.pre_fit_validation=}')

    if config.run.fit:
        trainer.fit(model=model, datamodule=datamodule)
    else:
        _logger.info(f'Skipping training as per {config.run.fit=}')

    if config.run.test:
        # `test` phase is supposed be run with the full validation dataset
        trainer.test(model=model, datamodule=datamodule, ckpt_path='best')
    else:
        _logger.info(f'Skipping testing as per {config.run.test=}')

    if config.run.predict:
        # `predict` phase is supposed be run with the full test dataset
        trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')
    else:
        _logger.info(f'Skipping predicting as per {config.run.predict=}')




if __name__ == '__main__':
    main()
