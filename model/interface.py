import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from typing import Union

import model as model_zoo
from utils import evaluation
from utils.config import ConfigDict


class PLModelInterface(pl.LightningModule):
    def __init__(self,
                 model: Union[str, ConfigDict],
                 criterion: Union[str, ConfigDict] = 'CrossEntropyLoss',
                 optimizer: Union[str, ConfigDict] = 'AdamW',
                 scheduler: Union[str, ConfigDict] = 'CosineAnnealingLR',
                 evaluator: str = 'accuracy',
                 **kwargs):
        super().__init__()

        model, criterion, optimizer, scheduler = map(
            lambda t: ConfigDict(type=t, cfg={}) if isinstance(t, str) else t,
            (model, criterion, optimizer, scheduler))

        self.model = getattr(model_zoo, model.type)(**model.cfg)
        self.criterion = getattr(nn, criterion.type)(**criterion.cfg)
        self.optimizer = getattr(optim, optimizer.type)
        self.optimizer_cfg = optimizer.cfg
        self.scheduler = getattr(optim.lr_scheduler, scheduler.type)
        self.scheduler_cfg = scheduler.cfg
        self.evaluate = getattr(evaluation, evaluator)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('val_loss', loss)

        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_acc', correct_num/len(out_digit), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_cfg)
        scheduler = self.scheduler(optimizer, **self.scheduler_cfg)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
