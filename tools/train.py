import sys

sys.path.append('.')  # run from project root

import os
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from tmpl import build_data_loaders, PLModelInterface


@hydra.main(version_base=None, config_path="../configs", config_name='config')
def main(cfg: DictConfig):
    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(OmegaConf.to_yaml(cfg))

    dls = build_data_loaders(cfg.data)
    model = PLModelInterface(**cfg.solver)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model, *dls)  # resume training by `ckpt_path='/path/to/checkpoint.ckpt'`


if __name__ == '__main__':
    main()
