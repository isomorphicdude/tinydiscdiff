from typing import Union

import torch
from torchvision.utils import make_grid


def broadcast_to(
    x: torch.Tensor, shape: Union[tuple[int, ...], list[int]]
) -> torch.Tensor:
    """Broadcasts a tensor to a given shape."""
    return x.view([x.shape[0]] + [1] * (len(shape) - len(x.shape))).expand(shape)


def loss2bpt(loss_dict, data_shape):
    """
    Normalises loss to bits per token.
    https://github.com/google-deepmind/md4/blob/main/md4/utils.py#L34
    """
    seq_len = torch.prod(torch.tensor(data_shape))
    rescale_to_bpd = 1.0 / (seq_len * torch.log(torch.tensor(2.0)))
    bpt_loss_dict = {}
    for k, v in loss_dict.items():
        if "loss" in k:
            bpt_loss_dict[k] = v * rescale_to_bpd
        else:
            bpt_loss_dict[k] = v
    return bpt_loss_dict


def vis_masked(
    x: torch.Tensor, mask_token: int, decode: Union[callable, None] = None
) -> Union[torch.Tensor, list[str]]:
    """
    Visualizes masked tokens in the input tensor by replacing
    them with a distinct value ([MASK] for text and red for images).
    """
    bs = x.shape[0]
    if x.dim() == 2:  # Text data
        vis = []
        for i in range(bs):
            vis.append(decode(x[i].tolist()))
    elif x.dim() == 4:  # Image data
        mask_any = x.eq(mask_token).any(dim=1, keepdim=True)
        vis = x.float().clone()
        mask_color = torch.tensor([1.0, 0.0, 0.0], device=vis.device).view(1, 3, 1, 1)
        vis = torch.where(mask_any.expand_as(vis), mask_color, vis)
        vis = make_grid(vis, nrow=8, normalize=True, value_range=(0, 1))
    return vis
