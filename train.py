import os
import math

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path, instantiate
import wandb
import torch
from torchvision.utils import make_grid, save_image

from src.models.md4 import MD4
from src.custom_datasets import get_data
from src.models.utils_model import vis_masked
from src.nets.utils_net import count_parameters_in_M
from src.trainers.utils_train import setup_logging

logger = setup_logging(__name__)


# TODO: write separate trainer class
def train_model(cfg, data_obj, model, device="cuda"):
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    def train_step(batch):
        bs = batch.shape[0]
        if cfg.train.antithetic:
            # evenly spaced timesteps in [0,1)
            t0 = torch.rand(1)
            t = torch.fmod(t0 + torch.arange(0.0, 1.0, step=1.0 / bs), 1.0)
        else:
            t = torch.rand(bs)
        t = t.to(device)
        loss = model.loss(batch, t, cond=None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_dict = {
            "loss": loss.item(),
        }
        return loss_dict

    eval_num = 16
    eval_latent = model.prior_sample(eval_num).to(device)

    @torch.no_grad()
    def eval_step(data_type):
        # only visualise samples
        x = model.sample(eval_num, sample_steps=100, cond=None, init=eval_latent)
        vocab = getattr(data_obj, "vocab", None)
        decode = getattr(vocab, "decode", None) if vocab is not None else None
        vis = vis_masked(x, mask_token=model.vocab_size, decode=decode)
        if data_type == "image":
            save_path = os.path.join(cfg.logging.ckpt_dir, f"sample_{counter}.png")
            save_image(vis, save_path)
        elif data_type == "text":
            print("Eval samples:")
            for i, txt in enumerate(vis):
                print(f"[Sample {i}]: {txt}")
            save_path = os.path.join(cfg.logging.ckpt_dir, f"sample_{counter}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                for i, txt in enumerate(vis):
                    f.write(f"[Sample {i}]: {txt}\n")
            f.close()

    num_iter = cfg.train.num_iter
    counter = 0
    while counter < num_iter:
        for batch in data_obj.train_loader:
            model.network.train()
            if isinstance(batch, (list, tuple)):
                label = batch[1]
                batch = batch[0]
                label = label.to(device)
            batch = batch.to(device)
            loss_dict = train_step(batch)
            counter += 1
            if counter % cfg.logging.log_freq == 0:
                print(
                    f"Iter {counter}: "
                    + ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                )
            if counter % cfg.logging.save_freq == 0:
                save_path = os.path.join(
                    cfg.logging.ckpt_dir, f"model_iter_{counter}.pt"
                )
                torch.save(model.network.state_dict(), save_path)
                print(f"Saved model checkpoint to {save_path}")
                # eval
                model.network.eval()
                eval_step(cfg.data.data_type)
            if counter >= num_iter:
                break


@hydra.main(config_path="configs", config_name=None, version_base="1.2")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    # create logging directory
    ckpt_dir = to_absolute_path(cfg.logging.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Logging to directory: {ckpt_dir}")
    cfg.logging.ckpt_dir = ckpt_dir
    device = cfg.device
    data_obj = get_data(cfg)
    network = instantiate(cfg.network)
    logger.info(f"Number of model parameters: {count_parameters_in_M(network)} M")
    model = MD4(
        network=network,
        vocab_size=data_obj.vocab_size,
        scheduler=cfg.model.scheduler,
        sample_shape=data_obj.sample_shape,
    )

    # set up
    model.network.to(device)
    train_model(cfg, data_obj, model, device=device)
    
if __name__ == "__main__":
    main()