#!/usr/bin/env python
import os
from pathlib import Path
import logging
import argparse
from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler
from lightning.pytorch import Trainer
from muonly.callbacks import PredictionWriter


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.resolve())

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
_logger = logging.getLogger(__name__)


def run(ckpt_file_path: Path, args_list: list[str]):
    torch.set_float32_matmul_precision("high")
    _logger.info(f"Setting torch float32 matmul precision to high")

    ckpt_file_path = ckpt_file_path.resolve()
    _logger.info(f"{ckpt_file_path=}")
    if not ckpt_file_path.exists():
        raise FileNotFoundError(f"Checkpoint file {ckpt_file_path} does not exist!")

    log_dir_path = ckpt_file_path.parents[1]
    if not log_dir_path.exists():
        raise FileNotFoundError(f"Log directory {log_dir_path} does not exist!")
    _logger.info(f"{log_dir_path=}")

    base_config = OmegaConf.load(log_dir_path / "config.yaml")

    cli_config = OmegaConf.from_cli(args_list)
    _logger.info(f'{cli_config=}')

    config = OmegaConf.merge(base_config, cli_config)

    _logger.info("instantiating model...")
    model = instantiate(config.model)

    _logger.info("instantiating datamodule...")
    datamodule = instantiate(config.datamodule)

    # NOTE: Callbacks
    output_dir_path = log_dir_path / "predict"
    output_dir_path.mkdir(exist_ok=True)
    output_file_path = output_dir_path / "test.h5"
    _logger.info(f"predictions will be saved to {output_file_path}")
    writer = PredictionWriter(output_file_path=output_file_path)

    callbacks = [
        writer,
    ]

    _logger.info("instantiating trainer...")
    trainer: Trainer = instantiate(config.trainer)(callbacks=callbacks)

    _logger.info("starting prediction...")
    trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_file_path, weights_only=False)
    _logger.info("prediction finished.")

    _logger.info("done")


def main():
    parser = argparse.ArgumentParser(description="Run app with OmegaConf")
    parser.add_argument(
        '-c', '--ckpt',
        dest='ckpt_file_path',
        type=Path,
        required=True,
        help='Path to the resolved YAML configuration file'
    )

    args, unknown_args = parser.parse_known_args()

    run(
        ckpt_file_path=args.ckpt_file_path,
        args_list=unknown_args
    )


if __name__ == "__main__":
    main()
