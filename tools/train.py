import sys
sys.path.append('.')  # run from project root

import argparse
import pytorch_lightning as pl

from yaml import load, Loader
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import data
from model import PLModelInterface
from utils.config import ConfigDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='config file path')
    return parser.parse_args()


def get_dls(train, val, batch_size=32, num_workers=4):
    return [
        DataLoader(getattr(data, s.TYPE)(**s.CFG),
                   batch_size=batch_size,
                   num_workers=num_workers,
                   shuffle=s is train)
        for s in (train, val)
    ]


def main(cfgs):
    dls = get_dls(**cfgs.DATA)
    model = PLModelInterface(**cfgs.SOLVER)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    ckpt_callback = ModelCheckpoint(save_top_k=2, monitor='val_acc', mode='max',
                                    filename=cfgs.SOLVER.MODEL.TYPE + '-{epoch}-{val_acc:.2f}')
    trainer = pl.Trainer(**cfgs.TRAINER, callbacks=[ckpt_callback, lr_monitor])
    trainer.fit(model, *dls)


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfgs = ConfigDict(load(f, Loader=Loader))
    main(cfgs)
