from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, OmegaConf

from .tabular_logger import TabularLogger


def pre_build_callbacks(cfg: DictConfig):
    if cfg.trainer.devices == 1 and cfg.trainer.get('strategy'):
        cfg.trainer.strategy = 'auto'

    output_dir = 'outputs'

    logger = [loggers.TensorBoardLogger(save_dir=output_dir, name=None)]
    callback = [
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.ModelCheckpoint(
            dirpath=logger[0].log_dir,
            filename='e{epoch}_acc{val_acc:.4f}',
            monitor='val/acc',
            save_last=True,
            mode='max',
            auto_insert_metric_name=False),
        callbacks.RichModelSummary(max_depth=1)
    ]

    if cfg.trainer.get('enable_progress_bar', True):
        callback.append(callbacks.RichProgressBar())
    else:
        logger.append(
            TabularLogger(save_dir=output_dir, name=None, version=logger[0].version))

    return cfg, dict(logger=logger, callbacks=callback)


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
