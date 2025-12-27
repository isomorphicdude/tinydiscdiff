"""
Implements https://arxiv.org/abs/2406.04329.
Simplified and Generalized Masked Diffusion for Discrete Data (Shi et al. 2024)
Based on: https://github.com/google-deepmind/md4/blob/main/md4/models/diffusion/md4.py
"""

from typing import Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.distributions as dis
from tqdm import tqdm

from .utils_model import broadcast_to
from .scheduler import MaskingSchedule


class MD4:
    def __init__(
        self,
        network: nn.Module,
        vocab_size: int,
        scheduler: str = "cosine",
        sample_shape: list[int] = [3, 32, 32],
    ):
        # super(MD4, self).__init__()
        self.network = network
        self.vocab_size = vocab_size
        self.scheduler = MaskingSchedule(schedule_fn_type=scheduler)
        # sample shape either (c,h,w) or (seq_len,)
        self.sample_shape = sample_shape if sample_shape is not None else [1]

    def forward_q(self, x_0, t):
        """Forward process q(x_t | x_0)."""
        alpha_t = self.scheduler.alpha(t)
        not_mask = torch.bernoulli(broadcast_to(alpha_t, x_0.shape)).to(x_0.device)
        return torch.where(not_mask.bool(), x_0, torch.full_like(x_0, self.vocab_size))

    def x0_pred(self, x_t, t):
        """Predicts E[x_0 | x_t]."""
        pass

    def xs_xt(self, x_t, t, s, cond=None):
        """Predicts p_theta(x_s | x_t)=q(x_s|x_t, x_0) for s < t."""
        bs = x_t.shape[0]
        alpha_t = self.scheduler.alpha(t)
        alpha_s = self.scheduler.alpha(s)
        # (bs, sample_shape, vocab_size) for logits
        logits, _ = self.network(x_t, t, cond=cond)
        x0_pred = torch.softmax(logits, dim=-1)

        mask_prob = (1 - alpha_s) / (1 - alpha_t)
        unmask_prob = (alpha_s - alpha_t) / (1 - alpha_t)
        # sample categorical
        probs_vocab = x0_pred * broadcast_to(unmask_prob, x0_pred.shape)
        probs_mask = torch.ones(
            [bs] + self.sample_shape + [1], device=x_t.device
        ) 
        probs_mask *= broadcast_to(mask_prob, probs_mask.shape)
        probs = torch.cat([probs_vocab, probs_mask], dim=-1)
        to_unmask = dis.Categorical(probs=probs).sample()
        # only replace masked positions
        x_s = torch.where(x_t == self.vocab_size, to_unmask, x_t)
        return x_s

    def prior_sample(self, batch_size):
        # starting from all masked
        return self.vocab_size * torch.ones(
            [batch_size] + self.sample_shape, dtype=torch.int32
        )

    def decode_x0(self, x0, cond=None):
        """Decodes remaining masked tokens after sampling."""
        # assumes x0 has shape (bs, sample_shape) consisting of vocab indices (tokens)
        masked = x0 == self.vocab_size
        x0_cliped = torch.where(masked, torch.zeros_like(x0), x0)
        masked = masked[..., None]
        t0 = torch.zeros(x0.shape[0], device=x0.device)
        logits, _ = self.network(x0, t0, cond=cond)
        probs = torch.where(
            masked,
            torch.softmax(logits, dim=-1),
            torch.nn.functional.one_hot(x0_cliped, self.vocab_size),
        )
        n_indep_axes = probs.ndim - 2
        dist = dis.Categorical(probs=probs.reshape(-1, self.vocab_size))
        return dist.sample().reshape(probs.shape[:-1])

    def sample(
        self,
        batch_size: int,
        sample_steps: int = 100,
        init: Union[torch.Tensor, None] = None,
        cond=None,
        return_list=False,
    ):
        """Samples from the model."""
        if init is not None:
            x = init
        else:
            x = self.prior_sample(batch_size)
        ret = [x]
        for i in tqdm(range(sample_steps), desc="Sampling"):
            t = (sample_steps - i) / sample_steps
            s = t - 1 / sample_steps
            if self.scheduler != "cosine":
                # for linear schedules (see Sec.4 in the paper)
                t = math.cos(math.pi / 2.0 * (1.0 - t)) * torch.ones(batch_size)
                s = math.cos(math.pi / 2.0 * (1.0 - s)) * torch.ones(batch_size)
            else:
                t = t * torch.ones(batch_size)
                s = s * torch.ones(batch_size)
            # expand
            # t = broadcast_to(t, [batch_size] + self.sample_shape).to(x.device)
            # s = broadcast_to(s, [batch_size] + self.sample_shape).to(x.device)
            t = t.to(x.device)
            s = s.to(x.device)

            x = self.xs_xt(x, t, s, cond=cond)
            if return_list:
                ret.append(x)
        return ret if return_list else x

    def loss(self, x_0, t, cond=None):
        """Computes the training loss."""
        # assumes x_0 consists of vocab indices (tokens)
        x_t = self.forward_q(x_0, t)
        batch_size = x_0.shape[0]
        mask = (x_t == self.vocab_size).float()
        logits, _ = self.network(x_t, t, cond=cond)
        log_mean = torch.nn.functional.log_softmax(logits, dim=-1)
        x0_onehot = torch.nn.functional.one_hot(x_0, self.vocab_size)
        # E_{q}[x0^T log \mu_theta(x_t, t)]
        neg_ce = torch.where(x0_onehot.bool(), x0_onehot * log_mean, 0.0)
        neg_ce = torch.sum(neg_ce, dim=-1)

        masked_neg_ce = torch.sum(mask * neg_ce, dim=list(range(len(x_0.shape))[1:]))
        # (bs,)
        loss = self.scheduler.dgamma_times_alpha(t) * masked_neg_ce

        return loss.mean()
