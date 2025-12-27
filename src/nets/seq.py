import math

import torch
import torch.nn as nn

from .utils_net import _timestep_embedding

class _AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ada_ln_attn = _AdaLayerNorm(dim, dim)
        self.ada_ln_mlp = _AdaLayerNorm(dim, dim)
        self.resid_dropout = nn.Dropout(dropout)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        attn_in = self.ada_ln_attn(x, time_cond)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.resid_dropout(attn_out)

        mlp_in = self.ada_ln_mlp(x, time_cond)
        x = x + self.mlp(mlp_in)
        return x


class SmallTransformer(nn.Module):
    """
    Small bidirectional Transformer for discrete diffusion (e.g. MD4 / MaskGIT-style).

    Extra mask token at index vocab_size.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        time_scale: float = 1_000.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")

        self.vocab_size = int(vocab_size)
        self.mask_token = int(vocab_size)
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.time_scale = float(time_scale)

        self.token_emb = nn.Embedding(self.vocab_size + 1, self.dim)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.SiLU(),
            nn.Linear(4 * self.dim, self.dim),
        )

        self.blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    dim=self.dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        # 
        self.norm = nn.LayerNorm(self.dim)
        # output head
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

        # weights init
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, *, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict | None]:
        #TODO: cond is kept as None for now (used for images)
        batch_size = x.shape[0]
        sample_shape = tuple(x.shape[1:])
        seq_len = math.prod(sample_shape) if len(sample_shape) > 0 else 1
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        tokens = x.reshape(batch_size, seq_len).long()
        device = tokens.device

        pos_ids = torch.arange(seq_len, device=device)
        h = self.token_emb(tokens) + self.pos_emb(pos_ids)[None, :, :]
        h = self.emb_dropout(h)

        t_emb = _timestep_embedding(t, self.dim, time_scale=self.time_scale).to(
            dtype=h.dtype
        )
        time_cond = self.time_mlp(t_emb)

        if cond is not None:
            if cond.shape != time_cond.shape:
                raise ValueError(
                    f"cond must have shape {time_cond.shape}, got {cond.shape}"
                )
            time_cond = time_cond + cond.to(device=device, dtype=time_cond.dtype)

        for block in self.blocks:
            h = block(h, time_cond)

        h = self.norm(h)
        logits = self.lm_head(h)
        logits = logits.view(batch_size, *sample_shape, self.vocab_size)
        return logits, None
