from torch.utils.data import DataLoader
from torchvision import transforms as T

from . import datasets


def build_data_loaders(cfg):
    return [
        DataLoader(getattr(datasets, cfg.datasets.type)(**cfgs),
                   **cfg.loader,
                   shuffle=split == 'train')
        for split, cfgs in cfg.datasets.splits.items()
    ]
