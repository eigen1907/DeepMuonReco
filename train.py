#!/usr/bin/env python
import logging
from pathlib import Path
import warnings
from aim.pytorch_lightning import AimLogger
import hydra
from hydra.utils import instantiate
import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning import seed_everything
from omegaconf import DictConfig
from omegaconf import OmegaConf
from coolname.impl import generate_slug
from tqdm import TqdmExperimentalWarning
from muonly.nn.utils import init_params
from muonly.utils.logging import log_everything


_logger = logging.getLogger(__name__)

# FIXME: config?
for logger_name in ["lightning", "matplotlib", "PIL", "aim", "filelock", "fsspec"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

OmegaConf.register_new_resolver(
    "slug",
    lambda pattern=2: generate_slug(pattern=pattern),
    use_cache=True,
    replace=True,
)

OmegaConf.register_new_resolver(
    name="eval",
    resolver=eval,
)


@hydra.main(
    config_path="./config",
    config_name="tts",
    version_base=None,
)
def main(config: DictConfig):
    output_dir = Path(config.paths.run_dir)
    with open(output_dir / "config.yaml", "w") as stream:
        OmegaConf.save(config=config, f=stream)

    torch.set_num_threads(config.torch.num_threads)
    torch.set_num_interop_threads(config.torch.num_interop_threads)
    torch.set_float32_matmul_precision(precision=config.torch.float32_matmul_precision)

    seed_everything(seed=config.run.seed, workers=True)

    model = instantiate(config.model)
    _logger.debug(f"{model=}")

    _logger.info("Initializing model weights")
    model.apply(init_params)

    callback_dict = instantiate(config.callbacks)
    logger = instantiate(config.logger)

    trainer: Trainer = instantiate(config.trainer)(
        callbacks=list(callback_dict.values()),
        logger=logger,
    )
    datamodule: LightningDataModule = instantiate(config.datamodule)

    if isinstance(logger, AimLogger):
        log_everything(logger=logger, config=config, model=model, output_dir=output_dir)

    if config.run.pre_fit_validation:
        # `validate` phase is supposed be run with the partial or full validation dataset
        trainer.validate(model=model, datamodule=datamodule)
    else:
        _logger.info(
            f"Skipping pre-fit validation as per {config.run.pre_fit_validation=}"
        )

    if config.run.fit:
        trainer.fit(model=model, datamodule=datamodule)
    else:
        _logger.info(f"Skipping training as per {config.run.fit=}")

    if config.run.test:
        # `test` phase is supposed be run with the full validation dataset
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best", weights_only=False)
    else:
        _logger.info(f"Skipping testing as per {config.run.test=}")

    if config.run.predict:
        # `predict` phase is supposed be run with the full test dataset
        trainer.predict(model=model, datamodule=datamodule, ckpt_path="best", weights_only=False)
    else:
        _logger.info(f"Skipping predicting as per {config.run.predict=}")


if __name__ == "__main__":
    main()
