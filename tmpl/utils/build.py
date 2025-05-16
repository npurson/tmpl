from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import callbacks, loggers
from omegaconf import DictConfig, OmegaConf

from .console_logger import ConsoleLogger


def build_callbacks(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    logger = [loggers.TensorBoardLogger(save_dir=output_dir, name=None, version='')]
    callback = [
        callbacks.LearningRateMonitor(logging_interval='step'),
        callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename='e{epoch}_acc{val/acc:.4f}',
            monitor='val/acc',
            mode='max',
            save_last=True,
            auto_insert_metric_name=False),
        callbacks.RichModelSummary(max_depth=2)
    ]

    if cfg.trainer.get('enable_progress_bar', True):
        callback.append(callbacks.RichProgressBar())
    else:
        logger.append(ConsoleLogger(save_dir=output_dir, name=None, version=''))
    return dict(logger=logger, callbacks=callback)


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
