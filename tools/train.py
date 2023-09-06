import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from tmpl import LitModule, build_data_loaders, pre_build_callbacks


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = pre_build_callbacks(cfg)

    dls = build_data_loaders(cfg.data)
    model = LitModule(**cfg)
    trainer = L.Trainer(**cfg.trainer, **callbacks)
    trainer.fit(model, *dls)


if __name__ == '__main__':
    main()
