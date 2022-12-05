import sys

sys.path.append('.')  # run from project root

import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tmpl import data
from tmpl.model import PLModelInterface


def get_dls(train, val, batch_size=32, num_workers=4):
    return [
        DataLoader(getattr(data, s.TYPE)(**s.CFG),
                   batch_size=batch_size,
                   num_workers=num_workers,
                   shuffle=s is train) for s in (train, val)
    ]


# TODO: Future Hydra versions will no longer change working directory at job runtime by default.
# See https://hydra.cc/docs/next/upgrades/1.1_to_1.2/changes_to_job_working_dir/ for more information.
@hydra.main(version_base=None, config_path="../configs", config_name='config')
def main(cfg: DictConfig):
    dls = get_dls(**cfg.DATA)
    model = PLModelInterface(**cfg.SOLVER)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_callback = ModelCheckpoint(save_top_k=2,
                                    monitor='val_acc',
                                    mode='max',
                                    filename=cfg.SOLVER.MODEL.TYPE +
                                    '-{epoch}-{val_acc:.2f}')
    trainer = pl.Trainer(**cfg.TRAINER, callbacks=[ckpt_callback, lr_monitor])
    trainer.fit(model, *dls)


if __name__ == '__main__':
    main()
