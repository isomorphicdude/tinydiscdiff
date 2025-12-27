# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""UNet implementation.

Adapted from https://github.com/google-research/vdm/blob/main/model_vdm.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .utils_net import _timestep_embedding


def _default_num_groups(channels: int, max_groups: int = 32) -> int:
    if channels <= 0:
        raise ValueError(f"channels must be > 0, got {channels}")
    max_groups = min(int(max_groups), int(channels))
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            print(f"Setting num_groups to {g} for {channels} channels")
            return g
    print("Setting num_groups to 1")
    return 1


class SelfAttention(nn.Module):
    """Self attention layer in UNets."""

    def __init__(self, channels: int, *, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        self.norm = nn.GroupNorm(_default_num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape (B,C,H,W), got {x.shape}")
        batch_size, channels, height, width = x.shape
        z = self.norm(x)
        z = z.flatten(2).transpose(1, 2)  # (B, HW, C)
        z, _ = self.attn(z, z, z, need_weights=False)
        z = z.transpose(1, 2).reshape(batch_size, channels, height, width)
        return x + z


class ResBlock(nn.Module):
    """Residual block in UNets."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        cond_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.cond_dim = int(cond_dim)

        self.norm1 = nn.GroupNorm(
            _default_num_groups(self.in_channels), self.in_channels
        )
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=3, padding=1
        )

        self.cond_proj = nn.Linear(self.cond_dim, self.out_channels, bias=False)
        nn.init.zeros_(self.cond_proj.weight)

        self.norm2 = nn.GroupNorm(
            _default_num_groups(self.out_channels), self.out_channels
        )
        self.dropout = nn.Dropout(float(dropout_rate))
        self.conv2 = nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=3, padding=1
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = (
            nn.Identity()
            if self.in_channels == self.out_channels
            else nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        cond_act = self.cond_proj(cond).to(dtype=h.dtype)
        h = h + cond_act[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return self.skip(x) + h


class UNet(nn.Module):
    """Time-conditioned UNet that predicts per-token logits for images."""

    def __init__(
        self,
        vocab_size: int = 256,
        *,
        in_channels: int = 3,
        d_channels: int = 32,
        n_layers: int = 2,
        num_attn_heads: int = 1,
        dropout_rate: float = 0.1,
        time_scale: float = 1_000.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token = vocab_size
        self.in_channels = in_channels
        self.d_channels = d_channels
        self.n_layers = n_layers
        self.time_scale = time_scale

        self.token_emb = nn.Embedding(self.vocab_size + 1, self.d_channels)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_channels, 4 * self.d_channels),
            nn.SiLU(),
            nn.Linear(4 * self.d_channels, self.d_channels),
        )

        self.in_conv = nn.Conv2d(
            self.d_channels * self.in_channels,
            self.d_channels,
            kernel_size=3,
            padding=1,
        )

        self.down_blocks = nn.ModuleList(
            [
                ResBlock(
                    self.d_channels,
                    self.d_channels,
                    cond_dim=self.d_channels,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.mid_block1 = ResBlock(
            self.d_channels,
            self.d_channels,
            cond_dim=self.d_channels,
            dropout_rate=dropout_rate,
        )
        self.mid_attn = SelfAttention(
            self.d_channels, num_heads=num_attn_heads, dropout=dropout_rate
        )
        self.mid_block2 = ResBlock(
            self.d_channels,
            self.d_channels,
            cond_dim=self.d_channels,
            dropout_rate=dropout_rate,
        )

        self.up_blocks = nn.ModuleList(
            [
                ResBlock(
                    2 * self.d_channels,
                    self.d_channels,
                    cond_dim=self.d_channels,
                    dropout_rate=dropout_rate,
                )
                for _ in range(self.n_layers + 1)
            ]
        )

        self.out_norm = nn.GroupNorm(
            _default_num_groups(self.d_channels), self.d_channels
        )
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            self.d_channels,
            self.in_channels * self.vocab_size,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        bs, nch, h, w = x.shape
        z = self.token_emb(x)  # (B,C,H,W,D)
        z = z.permute(0, 1, 4, 2, 3).view(bs, int(nch * self.d_channels), h, w)
        z = self.in_conv(z)

        t_emb = _timestep_embedding(t, self.d_channels, time_scale=self.time_scale).to(
            device=z.device
        )
        time_cond = self.time_mlp(t_emb)

        if cond is not None:
            time_cond = time_cond + cond.to(
                device=time_cond.device, dtype=time_cond.dtype
            )

        # save outputs for residual
        zs = [z]
        for block in self.down_blocks:
            z = block(z, time_cond)
            zs.append(z)
        z = self.mid_block1(z, time_cond)
        z = self.mid_attn(z)
        z = self.mid_block2(z, time_cond)

        for block in self.up_blocks:
            skip = zs.pop()
            z = torch.cat([z, skip], dim=1)
            z = block(z, time_cond)

        assert len(zs) == 0, "Mismatch in UNet skip connections."
        logits = self.out_conv(self.out_act(self.out_norm(z)))  # (B, C*V, H, W)
        logits = (
            logits.view(bs, nch, self.vocab_size, h, w)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        return logits, None
