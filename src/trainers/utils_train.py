"""Training utils for LSGM model."""
import os
import types
import random
import logging

import wandb
import numpy as np
import torch
import torch.distributed as dist


def setup_logging(name) -> logging.Logger:
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


logger = setup_logging(__name__)


def sort_ckpts(ckpt_dir: str, name: str) -> str | None:
    """
    Returns the latest checkpoint path for files named like `{name}_<step>.pt`.
    Supports names that contain underscores (e.g. `model_iter_5000.pt`).
    """
    if not os.path.isdir(ckpt_dir):
        logger.warning(f"Checkpoint directory not found: {ckpt_dir}")
        return None

    candidates: list[tuple[int, str]] = []
    prefix = f"{name}_"
    for fname in os.listdir(ckpt_dir):
        if not fname.startswith(prefix) or not fname.endswith(".pt"):
            continue
        step_str = fname.rsplit(".", 1)[0].split("_")[-1]
        try:
            step = int(step_str)
        except ValueError:
            continue
        candidates.append((step, fname))

    if not candidates:
        logger.warning(f"No checkpoints found in {ckpt_dir} for prefix '{prefix}'.")
        return None

    _, fname = max(candidates, key=lambda x: x[0])
    return os.path.join(ckpt_dir, fname)

def get_wandb_tab_text(path: str) -> wandb.Table:
        """Parse `sample_*.txt` produced by `_save_eval` into a per-sample W&B table."""
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        rows = []
        for line in raw.splitlines():
            if line.startswith("[Sample ") and "]:" in line:
                prefix, text = line.split(":", 1)
                sample_id = prefix.removeprefix("[Sample ").removesuffix("]").strip()
                rows.append([sample_id, text.lstrip()])

        if not rows:
            rows = [[0, raw]]

        return wandb.Table(data=rows, columns=["sample_id", "text"])


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def symmetrize_image_data(images):
    return 2.0 * images - 1.0


def unsymmetrize_image_data(images):
    return (images + 1.0) / 2.0


def normalize_symmetric(images):
    """
    Normalize images by dividing the largest intensity. Used for visualizing the intermediate steps.
    """
    b = images.shape[0]
    m, _ = torch.max(torch.abs(images).view(b, -1), dim=1)
    images /= m.view(b, 1, 1, 1) + 1e-3

    return images


def average_gradients(params, is_distributed):
    """Gradient averaging."""
    if not is_distributed:
        return

    if isinstance(params, types.GeneratorType):
        params = list(params)

    grad_params = [p for p in params if p.requires_grad and p.grad is not None]

    if not grad_params:
        return

    try:
        world_size = float(dist.get_world_size())

        for param in grad_params:
            # first sum all gradients
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            # then average them
            param.grad.data /= world_size

    except RuntimeError as e:
        print(f"Distributed gradient averaging failed: {e}")
        raise


def average_params(params, is_distributed):
    """parameter averaging."""
    if is_distributed:
        size = float(dist.get_world_size())
        for param in params:
            param.data /= size
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def broadcast_params(params, is_distributed):
    if is_distributed:
        for param in params:
            dist.broadcast(param.data, src=0)


def set_seed(seed=0, rank=0, use_distributed=False):
    """
    Sets the random seed for reproducibility in distributed training.
    """
    process_seed = seed + rank

    random.seed(process_seed)

    np.random.seed(process_seed)

    torch.manual_seed(process_seed)
    torch.cuda.manual_seed(process_seed)
    if use_distributed:
        torch.cuda.manual_seed_all(process_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
