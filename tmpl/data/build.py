from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .. import build_from_configs
from . import datasets


def build_data_loaders(cfg: DictConfig):
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    split_cfgs = cfg.datasets.pop('splits')

    if cfg.datasets.type in ('CIFAR10', 'CIFAR100', 'MNIST'):
        print(f'It seems you are using the {cfg.datasets.type} dataset from our demo config, '
              'we automatically add ToTensor() to the pipeline only for demo. It is usually '
              'unnecessary for your own dataset and you can modify this part as per your '
              'requirements.')
        split_cfgs = OmegaConf.to_container(split_cfgs)
        for s in split_cfgs:
            split_cfgs[s]['transform'] = T.ToTensor()

    return [
        DataLoader(
            build_from_configs(datasets, dict(**cfg.datasets, **cfgs)),
            **cfg.loader,
            shuffle=split == 'train') for split, cfgs in split_cfgs.items()
    ]
