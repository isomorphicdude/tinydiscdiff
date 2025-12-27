import math

import torch


def _timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    max_period: int = 10_000,
    time_scale: float = 1_000.0,
) -> torch.Tensor:
    """Sinusoidal timestep embedding for (batch,) timesteps in [0, 1]."""
    if timesteps.dim() != 1:
        raise ValueError(
            f"Expected timesteps to have shape (batch,), got {timesteps.shape}"
        )
    timesteps = timesteps.float() * time_scale

    half = dim // 2
    if half == 0:
        return torch.zeros(
            (timesteps.shape[0], dim), device=timesteps.device, dtype=timesteps.dtype
        )
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, device=timesteps.device, dtype=timesteps.dtype)
        / half
    )
    args = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb