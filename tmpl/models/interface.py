import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import lightning.pytorch as pl

from .. import models


class PLModelInterface(pl.LightningModule):

    def __init__(self, model, criterion, optimizer, scheduler, **kwargs):
        super().__init__()
        self.model = getattr(models, model.type)(**model.cfgs)
        self.criterion = getattr(nn, criterion.type)()
        self.optimizer = optimizer
        self.scheduler = scheduler

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
        # When you call self.log inside the validation_step and test_step,
        # Lightning automatically accumulates the metric and averages it
        # once it’s gone through the whole split (epoch).
        self.log('val_loss', loss)
        self.log('val_acc', self.evaluate(pred, y), prog_bar=True)
        return pred.argmax(dim=1), y

    def validation_epoch_end(self, step_outputs):
        preds = torch.cat([out[0] for out in step_outputs])
        targets = torch.cat([out[1] for out in step_outputs])

        preds_list = [
            torch.zeros(preds.shape, dtype=preds.dtype, device=preds.device)
            for _ in range(dist.get_world_size())
        ]
        targets_list = [
            torch.zeros(targets.shape, dtype=targets.dtype, device=targets.device)
            for _ in range(dist.get_world_size())
        ]
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
        optimizer = getattr(optim, self.optimizer.type)(self.parameters(), **self.optimizer.cfgs)
        scheduler = getattr(optim.lr_scheduler, self.scheduler.type)(optimizer,
                                                                     **self.scheduler.cfgs)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
