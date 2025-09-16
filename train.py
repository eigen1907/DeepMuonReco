#!/usr/bin/env python
from socket import gethostname
from getpass import getuser
import sys
from logging import getLogger
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
    config_name='config',
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

    torch.set_num_threads(config.num_threads)
    torch.set_float32_matmul_precision(precision=config.float32_matmul_precision)

    if config.seed:
        seed_everything(config.seed, workers=True)

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
    datamodule: LightningDataModule = instantiate(config.data)

    if isinstance(logger, AimLogger):
        logger.experiment.name = config.run_name

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
        for tag in config.tags:
            logger.experiment.add_tag(tag)

        description_file = output_dir / 'description.txt'
        if description_file.exists():
            _logger.info(f'Loading description from {description_file}')
            with open(description_file, 'r') as stream:
                description = stream.read()
            _logger.info(f'{description=}')
            logger.experiment.description = description
        elif description := config.get('description', None):
            logger.experiment.description = description


    if config.pre_fit_validation:
        trainer.validate(model=model, datamodule=datamodule)

    trainer.fit(model=model, datamodule=datamodule)

    # NOTE: `test` phase is supposed be run with the validation dataset
    test_output = trainer.test(model=model, datamodule=datamodule, ckpt_path='best')

    if config.predict:
        trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')

    if objective_name := config.get('optuna_objective', None):
        if len(test_output) != 1:
            _logger.warning(
                f'Expected a single metric dict, got {len(test_output)}: {test_output}'
            )
        metric_dict: dict = test_output[0]

        _logger.info(f'Setting optuna objective: {objective_name}')
        objective = metric_dict[objective_name]
    else:
        objective = None

    return objective




if __name__ == '__main__':
    main()
