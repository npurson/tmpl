import os

from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, open_dict

from .tabular_logger import TabularLogger


def pre_build_callbacks(cfg: DictConfig):
    # TODO: build callbacks based on configs
    # del cfg.trainer.xxx
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = None
    with open_dict(cfg):
        cfg.trainer.enable_progress_bar = False

    output_dir = 'outputs'
    cbs = {
        'logger': [
            loggers.TensorBoardLogger(save_dir=output_dir, name=None),
            TabularLogger(save_dir=output_dir, name=None)
        ],
        'callbacks': [
            callbacks.LearningRateMonitor(logging_interval='step'),
            callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, cfg.save_dir)
                if cfg.get('save_dir') else output_dir,
                filename='e{epoch}_acc{val_acc:.4f}',
                monitor='val_acc',
                save_last=True,
                mode='max',
                auto_insert_metric_name=False),
            callbacks.RichModelSummary(max_depth=-1),
            callbacks.RichProgressBar()
        ]
    }
    return cfg, cbs


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
