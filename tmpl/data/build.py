from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .. import build_from_configs
from . import datasets


def build_data_loaders(cfg: DictConfig):
    with open_dict(cfg):
        split_cfgs = cfg.datasets.pop('splits')
    return [
        DataLoader(
            build_from_configs(datasets, dict(**cfg.datasets, **cfgs)),
            **cfg.loader,
            shuffle=split == 'train') for split, cfgs in split_cfgs.items()
    ]
