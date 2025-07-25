from hydra.core.hydra_config import HydraConfig
from lightning import pytorch as pl
from omegaconf import DictConfig, OmegaConf

from .console_logger import ConsoleLogger


def build_callbacks(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    loggers = [pl.loggers.TensorBoardLogger(save_dir=output_dir, name=None, version='')]
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir,
            filename=f'e{{epoch}}_acc{{val/{cfg.evaluator.type}:.4f}}',
            monitor=f'val/{cfg.evaluator.type}',
            mode='max',
            save_last=True,
            auto_insert_metric_name=False),
        pl.callbacks.RichModelSummary(max_depth=2)
    ]

    if cfg.trainer.get('enable_progress_bar', True):
        callbacks.append(pl.callbacks.RichProgressBar())
    else:
        loggers.append(ConsoleLogger(save_dir=output_dir, name=None, version=''))
    return callbacks, loggers


def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)
