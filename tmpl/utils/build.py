import os
import lightning.pytorch as pl
from omegaconf import DictConfig, open_dict

from .tabular_logger import TabularLogger


def build_from_configs(cfg: DictConfig):
    # TODO: build callbacks based on configs
    # del cfg.trainer.xxx
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = None
    with open_dict(cfg):
        cfg.trainer.enable_progress_bar = False

    output_dir = 'outputs'
    callbacks = {
        'logger': [
            pl.loggers.TensorBoardLogger(save_dir=output_dir, name=None),
            TabularLogger(save_dir=output_dir, name=None)
        ],
        'callbacks': [
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, cfg.save_dir)
                if cfg.get('save_dir') else output_dir,
                filename='e{epoch}_acc{val_acc:.4f}',
                monitor='val_acc',
                save_last=True,
                mode='max',
                auto_insert_metric_name=False)
        ]
    }
    return cfg, callbacks
