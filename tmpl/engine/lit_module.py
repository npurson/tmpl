from collections import defaultdict

import lightning as L
import torch.nn as nn
import torch.optim as optim
from omegaconf import open_dict

from .. import build_from_configs, evaluation, models


class LitModule(L.LightningModule):

    def __init__(self,
                 model,
                 optimizer,
                 scheduler,
                 criterion=None,
                 evaluator=None,
                 **kwargs):
        super().__init__()
        self.model = build_from_configs(models, model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = build_from_configs(
            nn, criterion) if criterion else self.model.loss
        self.evaluator = build_from_configs(evaluation, evaluator)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, evaluator=None):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        if evaluator:
            evaluator.update(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        if isinstance(loss, dict):
            loss['loss'] = sum(loss.values())
            self.log_dict({f'{k}': v for k, v in loss.items()})
        else:
            self.log('loss', loss)
        return sum(loss.values()) if isinstance(loss, dict) else loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, self.evaluator)
        if self.evaluator:
            self.log(f'val/{self.evaluator.__class__.__name__}',
                     self.evaluator, sync_dist=True)

    def test_step(self, batch, batch_idx):
        self._step(batch, self.evaluator)
        if self.evaluator:
            self.log(f'test/{self.evaluator.__class__.__name__}',
                     self.evaluator, sync_dist=True)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer
        with open_dict(optimizer_cfg):
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        pg0 = []
        pgs = defaultdict(list)

        if paramwise_cfg:
            for pg_cfg in paramwise_cfg:
                pgs[pg_cfg.name].append(pg_cfg)

            for name, param in self.named_parameters():
                matched = False
                for pg_cfg in paramwise_cfg:
                    if 'name' in pg_cfg and pg_cfg.name in name:
                        pgs[pg_cfg.name].append(param)
                        matched = True
                        break
                if not matched:
                    pg0.append(param)
        else:
            pg0 = self.model.parameters()

        optimizer = build_from_configs(optim, optimizer_cfg, params=pg0)
        if paramwise_cfg:
            for pg in pgs.values():
                pg_cfg, *params = pg
                cfg = {}
                if 'lr_mult' in pg_cfg:
                    cfg['lr'] = optimizer_cfg.lr * pg_cfg.lr_mult
                optimizer.add_param_group({'params': params, **cfg})

        scheduler = build_from_configs(
            optim.lr_scheduler, self.scheduler, optimizer=optimizer)
        if 'interval' in self.scheduler:
            scheduler = {
                'scheduler': scheduler,
                'interval': self.scheduler.interval
            }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
