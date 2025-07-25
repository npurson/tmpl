from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .. import build_from_configs
from . import datasets


def build_data_loaders(cfg: DictConfig):
    if isinstance(cfg, DictConfig):
        with open_dict(cfg):
            split_cfgs = cfg.datasets.pop('splits')
    if isinstance(split_cfgs, ListConfig):
        split_cfgs = {split: {'split': split} for split in split_cfgs}

    if cfg.datasets.type in ('CIFAR10', 'MNIST'):
        print(
            'NOTE: For demonstration purposes, when using standard torchvision.datasets such as '
            f'{cfg.datasets.type}, we manually add ToTensor() here to ensure the pipeline is runnable. '
            'For typical use case with custom dataset, you should generally handle it elsewhere.'
        )
        split_cfgs = OmegaConf.to_container(split_cfgs)
        for s in split_cfgs:
            split_cfgs[s]['transform'] = T.ToTensor()

    return [
        DataLoader(
            build_from_configs(datasets, dict(**cfg.datasets, **cfgs)),
            **cfg.loader,
            shuffle=split == 'train') for split, cfgs in split_cfgs.items()
    ]
