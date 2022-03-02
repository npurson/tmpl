import sys
sys.path.append('.')  # run from project root

import argparse
import pytorch_lightning as pl

from yaml import load, Loader
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import data
from model import PLModelInterface
from utils.config import ConfigDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='config file path')
    return parser.parse_args()


def get_dls(train, val, batch_size=32, num_workers=4):
    return [
        DataLoader(getattr(data, s.TYPE)(**s.CFG, transform=transforms.ToTensor()),
                   batch_size=batch_size,
                   num_workers=num_workers,
                   shuffle=s is train)
        for s in (train, val)
    ]


def main(cfgs):
    dls = get_dls(**cfgs.DATA)
    model = PLModelInterface(**cfgs.MODEL, **cfgs.SOLVER)
    trainer = pl.Trainer(max_epochs=cfgs.SOLVER.NUM_EPOCHS,
                         precision=16,
                         gpus=1)
    trainer.fit(model, *dls)


if __name__ == '__main__':
    args = parse_args()
    with open(args.cfg, 'r') as f:
        cfgs = ConfigDict(load(f, Loader=Loader))
    main(cfgs)
