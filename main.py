import os

import wandb
import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.custom_datasets import get_data
from src.models.md4 import MD4
from src.nets.utils_net import count_parameters_in_M
from src.trainers.trainer import Trainer
from src.trainers.utils_train import setup_logging

logger = setup_logging(__name__)


@hydra.main(config_path="configs", config_name=None, version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    # wandb init
    if cfg.logging.use_wandb:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg)
        )

    ckpt_dir = to_absolute_path(cfg.logging.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg.logging.ckpt_dir = ckpt_dir
    logger.info(f"Logging to directory: {ckpt_dir}")

    device = cfg.device
    data_obj = get_data(cfg)
    network = instantiate(cfg.network)
    logger.info(f"Number of model parameters: {count_parameters_in_M(network)} M")

    model = MD4(
        network=network,
        vocab_size=data_obj.vocab_size,
        scheduler=cfg.model.scheduler,
        sample_shape=data_obj.sample_shape,
        device=device,
    )
    model.network.to(device)

    Trainer(cfg=cfg, data_obj=data_obj, model=model, device=device).run(resume=True)


if __name__ == "__main__":
    main()

