import os
import html
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchvision.utils import save_image
import wandb
from torchmetrics.image.fid import FrechetInceptionDistance

from src.models.utils_model import vis_masked
from .utils_train import setup_logging, sort_ckpts, get_wandb_tab_text

logger = setup_logging(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig, data_obj: Any, model: Any, device: str):
        self.cfg = cfg
        self.data_obj = data_obj
        self.model = model
        self.device = device

        self.data_type = str(cfg.data.data_type).lower()

        self.ckpt_dir = str(cfg.logging.ckpt_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.resume_iter = getattr(cfg.train, "resume_iter", None)
        self.ckpt_name = "model_iter"

        self.global_step = 0

        self.optimizer = torch.optim.AdamW(
            self.model.network.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        # common latent for eval samples
        self._eval_latent = self.model.prior_sample(cfg.eval.num_samples).to(
            self.device
        )

    def save_checkpoint(self, *, step: int | None = None) -> str:
        step = self.global_step if step is None else int(step)
        path = os.path.join(self.ckpt_dir, f"{self.ckpt_name}_{step}.pt")
        ckpt = {
            "global_step": step,
            "model": self.model.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, *, resume_iter: int | None = None) -> str | None:
        if resume_iter is None:
            resume_iter = self.resume_iter

        if resume_iter is not None:
            path = os.path.join(
                self.ckpt_dir, f"{self.ckpt_name}_{int(resume_iter)}.pt"
            )
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        else:
            path = sort_ckpts(self.ckpt_dir, self.ckpt_name)
            if path is None:
                return None

        ckpt = torch.load(path, map_location="cpu")
        self.model.network.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = int(ckpt.get("global_step", 0))
        logger.info(f"Loaded checkpoint: {path} (step={self.global_step})")
        return path

    def train_step(self, batch: Any) -> dict[str, float]:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(self.device)
        bs = batch.shape[0]
        if self.cfg.train.antithetic:
            t0 = torch.rand(1, device=self.device)
            t = (t0 + torch.arange(bs, device=self.device) / float(bs)) % 1.0
        else:
            t = torch.rand(bs, device=self.device)
        loss = self.model.loss(batch, t, cond=None)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        self.optimizer.step()
        return {"loss": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def _save_eval(self, x: torch.Tensor, *, step: int) -> str:
        vocab = getattr(self.data_obj, "vocab", None)
        decode = getattr(vocab, "decode", None) if vocab is not None else None
        vis = vis_masked(x, mask_token=self.model.vocab_size, decode=decode)

        if self.data_type == "image":
            save_path = os.path.join(self.ckpt_dir, f"sample_{step}.png")
            save_image(vis, save_path)
            return save_path

        samples_dir = os.path.join(self.ckpt_dir, f"samples_{step}")
        os.makedirs(samples_dir, exist_ok=True)

        if not isinstance(vis, list):
            raise TypeError("Expected text samples to be a list of strings")

        for i, txt in enumerate(vis):
            sample_path = os.path.join(samples_dir, f"sample_{i}.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(str(txt).rstrip())
                f.write("\n")

        return samples_dir


    def _prepare_for_fid(self, x: torch.Tensor, *, as_uint8: bool) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"FID expects image tensors shaped (N,C,H,W), got {x.shape}"
            )

        x = x.detach()
        if not x.is_floating_point():
            x = x.float()
        # assumes pixel range [0,255]
        x = x.clamp(0.0, 255.0)

        if x.shape[1] == 1:
            # grayscale
            x = x.repeat(1, 3, 1, 1)

        if x.shape[-2:] != (299, 299):
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if as_uint8:
            return x.round().to(torch.uint8)
        return (x / 255.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def compute_fid(self) -> float | None:
        fid_every = self.cfg.eval.get("fid_every", None)
        if fid_every is None or self.global_step % int(fid_every) != 0:
            return None

        fid, expects_float = self._make_fid_metric()
        as_uint8 = not expects_float

        # true images
        n_real = 0
        for batch in self.data_obj.train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            imgs = self._prepare_for_fid(batch.to(self.device), as_uint8=as_uint8)
            fid.update(imgs, real=True)
            n_real += imgs.shape[0]
            if n_real >= self.cfg.eval.fid_num_samples:
                break

        # fake images
        n_fake = 0
        fid_bs = self.cfg.eval.fid_batch_size
        if fid_bs is None:
            raise ValueError("`eval.fid_bs` must be set to compute FID")
        fid_bs = int(fid_bs)
        while n_fake < self.cfg.eval.fid_num_samples:
            bs = min(fid_bs, int(self.cfg.eval.fid_num_samples) - n_fake)
            x_fake = self.model.sample(
                bs, sample_steps=self.cfg.eval.fid_sample_steps, cond=None
            )
            imgs = self._prepare_for_fid(x_fake.to(self.device), as_uint8=as_uint8)
            fid.update(imgs, real=False)
            n_fake += imgs.shape[0]

        return float(fid.compute().detach().cpu().item())

    def _make_fid_metric(self) -> tuple[FrechetInceptionDistance, bool]:
        try:
            fid = FrechetInceptionDistance(normalize=False)
            expects_float = False
        except TypeError:
            fid = FrechetInceptionDistance()
            expects_float = bool(getattr(fid, "normalize", False))

        fid = fid.to(self.device)
        return fid, expects_float

    @torch.no_grad()
    def evaluate(self) -> dict[str, Any]:
        self.model.network.eval()
        x = self.model.sample(
            self.cfg.eval.num_samples,
            sample_steps=self.cfg.eval.sample_steps,
            cond=None,
            init=self._eval_latent,
        )

        metrics: dict[str, Any] = {}
        artifact_path = self._save_eval(x, step=self.global_step)
        metrics["eval_artifact"] = artifact_path

        fid = self.compute_fid()
        if fid is not None:
            metrics["fid"] = fid
        return metrics

    def run(self, *, resume: bool = True) -> None:
        self.model.network.to(self.device)
        if resume:
            self.load_checkpoint()

        use_wandb = bool(self.cfg.logging.use_wandb)
        counter = self.global_step
        while counter < self.cfg.train.num_iter:
            for batch in self.data_obj.train_loader:
                self.model.network.train()
                loss_dict = self.train_step(batch)
                counter += 1
                self.global_step = counter
                wandb_payload: dict[str, Any] = {}

                if counter % self.cfg.logging.log_freq == 0:
                    logger.info(
                        f"Iter {counter}: "
                        + ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                    )
                    if use_wandb:
                        wandb_payload.update(loss_dict)

                if counter % self.cfg.logging.save_freq == 0:
                    ckpt_path = self.save_checkpoint(step=counter)
                    logger.info(f"Saved checkpoint: {ckpt_path}")

                    eval_dict = self.evaluate()
                    logger.info(
                        "Eval: "
                        + ", ".join([f"{k}: {v}" for k, v in eval_dict.items()])
                    )

                    if use_wandb:
                        wandb_payload.update(
                            {
                                k: v
                                for k, v in eval_dict.items()
                                if k != "eval_artifact"
                            }
                        )
                        if self.data_type == "image":
                            wandb_payload["samples"] = wandb.Image(
                                eval_dict["eval_artifact"],
                                caption=f"Samples at step {counter}",
                            )
                        elif self.data_type == "text":
                            wandb_payload["samples"] = get_wandb_tab_text(
                                eval_dict["eval_artifact"]
                            )

                if use_wandb and wandb_payload:
                    wandb.log(wandb_payload, step=counter)

                if counter >= self.cfg.train.num_iter:
                    break
