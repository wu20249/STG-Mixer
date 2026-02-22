import argparse
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import utils.metrics
import utils.losses
import pandas as pd

class SupervisedForecastTask(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        regressor="linear",
        loss="mse_with_regularizer",
        pre_len: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.5e-3,
        feat_max_val: float = 1.0,
        feat_min_val: float = 1.0,
        **kwargs
    ):
        super(SupervisedForecastTask, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.regressor = (
            nn.Linear(
                self.model.hyperparameters.get("hidden_dim")
                or self.model.hyperparameters.get("output_dim"),
                self.hparams.pre_len,
            )
            if regressor == "linear"
            else regressor
        )
        self._loss = loss
        self.feat_max_val = feat_max_val
        self.feat_min_val = feat_min_val
        self.adj_num_nodes = 47
        self.train_loss_each_epoch = []

    def forward(self, x):
        # (batch_size, seq_len, num_nodes)
        batch_size, _, num_nodes, _ = x.size()
        # (batch_size, num_nodes, hidden_dim)
        predictions = self.model(x)
        # (batch_size * num_nodes, hidden_dim)
        # (batch_size * num_nodes, pre_len)
        predictions = torch.where(predictions < 0, -predictions, predictions)
        return predictions

    def shared_step(self, batch, batch_idx):
        # (batch_size, seq_len/pre_len, num_nodes)
        x, y = batch
        num_nodes = self.adj_num_nodes
        predictions = self(x)
        predictions = predictions.transpose(1, 2).reshape((-1, num_nodes))
        y = y.reshape((-1, y.size(2)))
        return predictions, y

    def loss(self, inputs, targets):
        if self._loss == "mse":
            return F.mse_loss(inputs, targets)
        if self._loss == "mse_with_regularizer":
            return utils.losses.mse_with_regularizer_loss(inputs, targets, self)
        raise NameError("Loss not supported:", self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log("train_loss", loss)
        self.train_loss_each_epoch.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = self.min_max_denormalize(predictions)
        y = self.min_max_denormalize(y)

        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        r2 = utils.metrics.r2(predictions, y)
        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = self.min_max_denormalize(predictions)
        y = self.min_max_denormalize(y)

        rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(predictions, y))
        mae = torchmetrics.functional.mean_absolute_error(predictions, y)
        r2 = utils.metrics.r2(predictions, y)

        pre_len = int(self.hparams.pre_len)
        B = predictions.shape[0] // pre_len
        N = predictions.shape[1]

        pred_by_h = predictions.view(B, pre_len, N)
        y_by_h = y.view(B, pre_len, N)

        pred_last = pred_by_h[:, -1, :].reshape(-1, N)
        y_last = y_by_h[:, -1, :].reshape(-1, N)

        r2_last = utils.metrics.r2(pred_last, y_last)

        self.log("val_R2_last", r2_last, prog_bar=True, on_epoch=True, logger=True)

        self.log_dict(
            {"RMSE": rmse, "MAE": mae, "val_r2": r2,
             "RMSE_last": torch.sqrt(torch.mean((pred_last - y_last) ** 2)),
             "MAE_last": torch.mean(torch.abs(pred_last - y_last)),
             "R2_last": r2_last},
            prog_bar=True, on_epoch=True, logger=True
        )

        # loss
        loss = self.loss(predictions, y)
        self.log("val_loss", loss)

        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def min_max_denormalize(self, normalized_data):
        return normalized_data * (self.feat_max_val - self.feat_min_val) + self.feat_min_val


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        pre_len = int(self.hparams.pre_len)

        if pre_len >= 3:
            monitor_metric = "val_r2"
        else:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8,
            threshold=1e-3, cooldown=2, min_lr=1e-4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", "--lr", type=float, default=0.001)

        parser.add_argument("--weight_decay", "--wd", type=float, default=1.5e-3)
        parser.add_argument("--loss", type=str, default="mse_with_regularizer")
        return parser
