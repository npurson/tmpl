import lightning as L
import torch.nn as nn
import torch.optim as optim

from .. import build_from_configs, evaluation, models


class LitModule(L.LightningModule):

    def __init__(self, *, model, optimizer, scheduler, criterion=None, evaluator=None, **kwargs):
        super().__init__()
        self.model = build_from_configs(models, model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = build_from_configs(nn, criterion) if criterion else self.model.loss
        self.train_evaluator = build_from_configs(evaluation, evaluator)
        self.test_evaluator = build_from_configs(evaluation, evaluator)

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
        loss = self._step(batch, self.train_evaluator)
        self.log('train/loss', loss)
        if self.train_evaluator:
            self.log('train/acc', self.train_evaluator, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, 'test')

    def _shared_eval(self, batch, prefix):
        loss = self._step(batch, self.test_evaluator)
        # Lightning automatically accumulates the metric and averages it
        # if `self.log` is inside the `validation_step` and `test_step`
        self.log(f'{prefix}/loss', loss, sync_dist=True)
        if self.test_evaluator:
            self.log(f'{prefix}/acc', self.test_evaluator, sync_dist=True)

    def configure_optimizers(self):
        optimizer = build_from_configs(optim, self.optimizer, params=self.model.parameters())
        scheduler = build_from_configs(optim.lr_scheduler, self.scheduler, optimizer=optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
