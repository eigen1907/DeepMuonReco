#!/usr/bin/env python
from pathlib import Path
import argparse
import hydra
from hydra.utils import instantiate
import torch
from omegaconf import OmegaConf
from deepmuonreco.model import Model
from deepmuonreco.callbacks import PredictionWriter


OmegaConf.register_new_resolver(
    name='eval',
    resolver=eval,
)


def run(ckpt_file_path: Path, gpu_id: int):
    ckpt_file_path = ckpt_file_path.resolve()
    print(f'{ckpt_file_path=}')
    if not ckpt_file_path.exists():
        raise FileNotFoundError(f'Checkpoint file {ckpt_file_path} does not exist!')

    log_dir_path = ckpt_file_path.parents[1]
    if not log_dir_path.exists():
        raise FileNotFoundError(f'Log directory {log_dir_path} does not exist!')
    print(f'{log_dir_path=}')

    # NOTE: config_path in initialize() must be relative
    log_dir_rel = log_dir_path.relative_to(Path.cwd())
    hydra.initialize(
        config_path=str(log_dir_rel),
        version_base=None,
    )
    config = hydra.compose(config_name='config')

    device = torch.device(f'cuda:{gpu_id}')

    print('instantiating model...')
    model = Model.from_config(config=config)

    print(f'loading checkpoint from {ckpt_file_path}...')
    model.load_state_dict(
        torch.load(
            ckpt_file_path,
            weights_only=False,
            map_location=device
        )['state_dict']
    )
    print(f'moving model to {device}...')
    model = model.to(device)

    print('instantiating datamodule...')
    datamodule = instantiate(config.data)

    # NOTE: Callbacks
    output_dir_path = log_dir_path / 'predict'
    output_dir_path.mkdir(exist_ok=True)
    output_file_path = output_dir_path / 'test.h5'
    print(f'predictions will be saved to {output_file_path}')
    writer = PredictionWriter(output_file_path=output_file_path)

    callbacks = [
        writer,
    ]

    config.trainer.enable_progress_bar = True
    print('instantiating trainer...')
    trainer = hydra.utils.instantiate(config.trainer)(callbacks=callbacks)

    print('starting prediction...')
    trainer.predict(model=model, datamodule=datamodule)
    print('prediction finished.')

    print('done')


def main():
    parser = argparse.ArgumentParser(
        description='Run prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-c', '--ckpt', dest='ckpt_file_path', type=Path, required=True, help='Path to the checkpoint file')
    parser.add_argument('-g', '--gpu', dest='gpu_id', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    run(**vars(args))



if __name__ == '__main__':
    main()
