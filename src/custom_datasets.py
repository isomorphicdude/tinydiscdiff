"""
Implements custom datasets and data loaders for text and image data.
Based on: https://www.youtube.com/watch?v=kCc8FmEb1nY by Andrej Karpathy
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision

def get_data(cfg):
    data_type = cfg.data.data_type.lower()
    if data_type == "text":
        return get_toy_text(cfg)
    elif data_type == "image":
        dataset_name = getattr(cfg.data, "dataset_name", None)
        return get_image_dataset(cfg, dataset_name=dataset_name)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


@dataclass(frozen=True)
class TinyTextVocab:
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        # when i=vocab_size decodes to [MASK]
        decoded = []
        for i in ids:
            if i in self.itos:
                decoded.append(self.itos[i])
            else:
                decoded.append("[MASK]")
        return "".join(decoded)

    def __len__(self) -> int:
        return len(self.stoi)


@dataclass(frozen=True)
class TextDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    vocab: TinyTextVocab
    vocab_size: int
    sample_shape: list[int]


def get_toy_text(cfg) -> TextDataBundle:
    """
    Returns dataset and loaders for tiny Shakespeare dataset.
    """
    batch_size = cfg.data.batch_size
    text_path = cfg.data.data_path
    if text_path is None:
        raise ValueError("cfg.data_path must be specified for text dataset.")

    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    f.close()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab = TinyTextVocab(stoi=stoi, itos=itos)

    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    val_split = 0.1
    n_train = int((1.0 - val_split) * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]

    # sequence length to be modelled by the diffusion
    seq_len = cfg.model.seq_len

    train_dataset = TextDataset(train_data, seq_len)
    val_dataset = TextDataset(val_data, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return TextDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        vocab_size=cfg.data.vocab_size,
        sample_shape=[seq_len],
    )


@dataclass(frozen=True)
class ImageDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    vocab_size: int
    num_classes: int | None
    sample_shape: list[int]


def get_image_dataset(cfg, dataset_name: str | None = None) -> ImageDataBundle:
    """
    Returns dataset and loaders for torchvision image datasets.
    """
    vocab_size = cfg.data.vocab_size
    batch_size = cfg.data.batch_size

    dataset_name = dataset_name or getattr(cfg, "dataset_name", None)

    name = str(dataset_name).lower().replace("-", "").replace("_", "")
    data_dir = cfg.data.data_path

    to_tokens = get_image_transforms()

    dataset_cls = __IMAGE_DATASETS__[name]
    sample_shape = dataset_cls.sample_shape
    num_classes = dataset_cls.num_classes
    train_dataset = dataset_cls(root=data_dir, train=True, transform=to_tokens)
    val_dataset = dataset_cls(root=data_dir, train=False, transform=to_tokens)

    val_split = 0.1
    if val_split is not None:
        val_split = float(val_split)
        n_val = int(val_split * len(train_dataset))
        n_train = len(train_dataset) - n_val
        train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return ImageDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=vocab_size,
        num_classes=num_classes,
        sample_shape=sample_shape,
    )


def get_image_transforms(manual_transforms=None):
    if manual_transforms is not None:
        return manual_transforms
    transforms = torchvision.transforms

    return transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: x.long()),
        ]
    )


__IMAGE_DATASETS__ = {}


def register_image_dataset(name: str):
    def wrapper(cls):
        __IMAGE_DATASETS__[name] = cls
        return cls

    return wrapper


@register_image_dataset(name="mnist")
class MNISTDataset(torchvision.datasets.MNIST):
    sample_shape = [1, 28, 28]
    num_classes = 10

    def __init__(
        self, root: Path, train: bool = True, transform=None, target_transform=None
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )


@register_image_dataset(name="fashionmnist")
class FashionMNISTDataset(torchvision.datasets.FashionMNIST):
    sample_shape = [1, 28, 28]
    num_classes = 10

    def __init__(
        self, root: Path, train: bool = True, transform=None, target_transform=None
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )


@register_image_dataset(name="cifar10")
class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    sample_shape = [3, 32, 32]
    num_classes = 10

    def __init__(
        self, root: Path, train: bool = True, transform=None, target_transform=None
    ):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
        )


@register_image_dataset(name="svhn")
class SVHNDataset(torchvision.datasets.SVHN):
    sample_shape = [3, 32, 32]
    num_classes = 10

    def __init__(
        self, root: Path, train: bool = True, transform=None, target_transform=None
    ):
        if target_transform is None:
            target_transform = lambda label: 0 if label == 10 else int(label)
        super().__init__(
            root=root,
            split="train" if train else "test",
            transform=transform,
            target_transform=target_transform,
        )


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx : idx + self.seq_len]
