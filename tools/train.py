import os

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from tmpl import LitModule, build_callbacks, build_data_loaders


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    if os.environ.get('LOCAL_RANK', 0) == 0:
        print(OmegaConf.to_yaml(cfg))
    callbacks = build_callbacks(cfg)

    dls = build_data_loaders(cfg.data)
    model = LitModule(**cfg)
    trainer = L.Trainer(**cfg.trainer, **callbacks)
    trainer.fit(model, *dls)


if __name__ == '__main__':
    main()
