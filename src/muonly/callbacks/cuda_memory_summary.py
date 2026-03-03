from pathlib import Path
import torch
from lightning.pytorch.callbacks import Callback
from hydra.core.hydra_config import HydraConfig


class CUDAMemorySummary(Callback):
    """
    """

    def on_fit_end(self, trainer, pl_module) -> None:
        """
        """
        if not trainer.is_global_zero:
            return

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f'hydra:runtime.output_dir does not exist: {output_dir}')

        log_dir = output_dir / 'cuda_memory_summary'
        log_dir.mkdir(parents=True, exist_ok=True)

        for idx, device_id in enumerate(trainer.device_ids):
            device = torch.device(f'cuda:{device_id}')
            cuda_memory_summary = torch.cuda.memory_summary(device=device)

            if idx == 0:
                print(cuda_memory_summary)

            config_path = log_dir / f'{device_id}.txt'
            with open(config_path, 'w') as stream:
                stream.write(cuda_memory_summary)
