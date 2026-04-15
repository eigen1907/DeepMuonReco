#!/usr/bin/env python
import os
from pathlib import Path
import logging
import argparse
import hydra
from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf
from muonly.callbacks import PredictionWriter


if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = str(Path(__file__).parent.resolve())


OmegaConf.register_new_resolver(
    "slug",
    lambda pattern=2: None,
    use_cache=True,
    replace=True,
)


OmegaConf.register_new_resolver(
    name="len",
    resolver=len,
)

_logger = logging.getLogger(__name__)


def run(ckpt_file_path: Path, gpu_id: int, enable_progress_bar: bool = False):
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

    # NOTE: config_path in initialize() must be relative
    log_dir_rel = log_dir_path.relative_to(Path.cwd())
    hydra.initialize(
        config_path=str(log_dir_rel),
        version_base=None,
    )
    config = hydra.compose(config_name="config")


    device = torch.device(f"cuda:{gpu_id}")

    _logger.info("instantiating model...")
    model = instantiate(config.model)

    _logger.info(f"loading checkpoint from {ckpt_file_path}...")
    model.load_state_dict(
        torch.load(ckpt_file_path, weights_only=False, map_location=device)[
            "state_dict"
        ]
    )
    _logger.info(f"moving model to {device}...")
    model = model.to(device)

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
    config.trainer.enable_progress_bar = enable_progress_bar
    trainer = instantiate(config.trainer)(callbacks=callbacks)

    _logger.info("starting prediction...")
    trainer.predict(model=model, datamodule=datamodule)
    _logger.info("prediction finished.")

    _logger.info("done")


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        dest="ckpt_file_path",
        type=Path,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu_id", type=int, default=0, help="GPU id to use"
    )
    parser.add_argument(
        "-p",
        "--progress-bar",
        dest="enable_progress_bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to show progress bar during prediction",
    )
    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
