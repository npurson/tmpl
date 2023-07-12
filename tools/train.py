import sys

sys.path.append('.')  # run from project root

import os
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from tmpl import LitModel, build_data_loaders, build_from_configs


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    cfg, callbacks = build_from_configs(cfg)

    dls = build_data_loaders(cfg.data)
    model = LitModel(**cfg.solver)
    trainer = L.Trainer(**cfg.trainer, **callbacks)
    trainer.fit(model, *dls)  # resume training by `ckpt_path='/path/to/checkpoint.ckpt'`


if __name__ == '__main__':
    main()
