import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from .. import models
from .. import evaluation


class PLModelInterface(pl.LightningModule):

    def __init__(self, model, criterion, optimizer, scheduler, evaluator, **kwargs):
        super().__init__()
        self.model = getattr(models, model.type)(**model.cfgs)
        self.criterion = getattr(nn, criterion.type)()
        self.optimizer = optimizer
        self.scheduler = scheduler
        # TODO: separate the evaluator for train and valid
        self.evaluator = getattr(evaluation, evaluator.type)(**evaluator.cfgs)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, evaluate=True):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        if evaluate:
            self.evaluator(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss)
        # Refer to https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#logging-torchmetrics
        self.log('train_acc', self.evaluator, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, 'test')

    def _shared_eval(self, batch, prefix):
        loss = self._step(batch)
        # Lightning automatically accumulates the metric and averages it
        # if `self.log` is inside the `validation_step` and `test_step`
        self.log(f'{prefix}_loss', loss)
        self.log(f'{prefix}_acc', self.evaluator, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer.type)(self.parameters(), **self.optimizer.cfgs)
        scheduler = getattr(optim.lr_scheduler, self.scheduler.type)(optimizer,
                                                                     **self.scheduler.cfgs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
