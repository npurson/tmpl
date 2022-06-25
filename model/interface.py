import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
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

        # Convert str into ConfigDict
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

    def _step(self, batch, evaluate=True, return_pred=False):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        if return_pred:
            return loss, pred, y
        if evaluate:
            return loss, self.evaluate(pred, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self._step(batch, return_pred=True)
        self.log('val_loss', loss)
        self.log('val_acc', self.evaluate(pred, y), prog_bar=True)
        return pred.argmax(dim=1), y

    def validation_epoch_end(self, step_outputs):
        preds = torch.cat([out[0] for out in step_outputs])
        targets = torch.cat([out[1] for out in step_outputs])

        preds_list = [
            torch.zeros(preds.shape, dtype=preds.dtype, device=preds.device)
            for _ in range(dist.get_world_size())]
        targets_list = [
            torch.zeros(targets.shape, dtype=targets.dtype, device=targets.device)
            for _ in range(dist.get_world_size())]
        dist.all_gather(preds_list, preds)
        dist.all_gather(targets_list, targets)

        if self.trainer.is_global_zero:
            preds = torch.cat(preds_list)
            targets = torch.cat(targets_list)
            print('acc:', (preds == targets).float().mean().item())

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        self.log('test_loss', loss)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_cfg)
        scheduler = self.scheduler(optimizer, **self.scheduler_cfg)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
