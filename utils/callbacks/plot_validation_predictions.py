import numpy as np
import matplotlib.pyplot as plt
from utils.callbacks.base import BestEpochCallback
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import utils
import torchmetrics
import torchmetrics.functional as TMF

def myr2(y_true, y_pred):

    # Total Sum of Squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Regression Sum of Squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # R^2
    r2 = 1 - ss_res / ss_tot
    return r2

class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor="", mode="min"):
        super(PlotValidationPredictionsCallback, self).__init__(monitor=monitor, mode=mode)
        self.ground_truths = []
        self.predictions = []

    def on_fit_start(self, trainer, pl_module):
        self.ground_truths.clear()
        self.predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        predictions, y = outputs
        if len(self.predictions) > 0:
            last_r2 = utils.metrics.r2(self.predictions[0], self.ground_truths[0]).cpu().numpy()
            current_r2 = utils.metrics.r2(predictions, y).cpu().numpy()
            if (current_r2 < last_r2):
                 return

        self.ground_truths.clear()
        self.predictions.clear()
        print("********************************")
        print("********best epoch change*******")
        self.rows_indicator(y, predictions)
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y)
        self.predictions.append(predictions)

    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.ground_truths.clear()
        self.predictions.clear()
        predictions, y = outputs
        predictions = predictions.cpu().numpy()
        y = y.cpu().numpy()
        self.ground_truths.append(y)
        self.predictions.append(predictions)

    def on_test_end(self, trainer, pl_module):
        self.save_to_csv()

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if trainer.state.fn == 'fit':
            print("This is validation during fitting.")

        elif trainer.state.fn == 'validate':
            print("This is a standalone validation.")
            self.save_to_csv()

    def save_to_csv(self):
        y = self.ground_truths[0]
        predictions = self.predictions[0]
        print("will save to csv...")
        self.rows_indicator(y, predictions)

        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()

        y = y.reshape(-1, y.shape[-1])  # [T, 47]
        predictions = predictions.reshape(-1, predictions.shape[-1])

        directory_path = "/home/yn/cx/wzk/yuce/predic_log"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        max_number = 0
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                try:
                    number = int(filename.split(".")[0].split("_")[-1])
                    max_number = max(max_number, number)
                except:
                    pass
        max_number += 1

        pd.DataFrame(predictions).to_csv(
            os.path.join(directory_path, f"predictions_{max_number}.csv"),
            index=False, header=False
        )
        pd.DataFrame(y).to_csv(
            os.path.join(directory_path, f"ground_truth_{max_number}.csv"),
            index=False, header=False
        )
        print("saved to:", max_number)

    def rows_indicator(self, y, predictions):

        y_t = y.detach().cpu().float() if torch.is_tensor(y) else torch.as_tensor(y, dtype=torch.float32)
        p_t = predictions.detach().cpu().float() if torch.is_tensor(predictions) else torch.as_tensor(predictions,
                                                                                                      dtype=torch.float32)

        def _guess_hdim(t: torch.Tensor) -> int:
            if hasattr(self, "pre_len") and isinstance(self.pre_len, int) and self.pre_len in t.shape:
                return list(t.shape).index(self.pre_len)
            small = [i for i, s in enumerate(t.shape) if s <= 8]
            if t.ndim == 3 and small:
                return small[0]
            if t.ndim == 2:
                return int(t.shape[0] <= t.shape[1])
            return t.ndim - 1

        hdim = _guess_hdim(p_t)

        p_vec = p_t.reshape(-1)
        y_vec = y_t.reshape(-1)
        mse = TMF.mean_squared_error(p_vec, y_vec)
        rmse = torch.sqrt(mse)
        mae = TMF.mean_absolute_error(p_vec, y_vec)
        r2 = TMF.r2_score(p_vec, y_vec)

        p_last = p_t.select(hdim, p_t.shape[hdim] - 1).reshape(-1)
        y_last = y_t.select(hdim, y_t.shape[hdim] - 1).reshape(-1)
        mse_last = TMF.mean_squared_error(p_last, y_last)
        rmse_last = torch.sqrt(mse_last)
        mae_last = TMF.mean_absolute_error(p_last, y_last)
        r2_last = TMF.r2_score(p_last, y_last)

        y_np = y_t.numpy()
        pred_np = p_t.numpy()

        if not hasattr(self, "buffer_metrics"):
            self.buffer_metrics = []
        self.buffer_metrics.append({
            "RMSE": float(rmse.item()),
            "MAE": float(mae.item()),
            "R2": float(r2.item()),
            "RMSE_last": float(rmse_last.item()),
            "MAE_last": float(mae_last.item()),
            "R2_last": float(r2_last.item()),
        })

        if not hasattr(self, "buffer_arrays"):
            self.buffer_arrays = []
        self.buffer_arrays.append({
            "y": y_np,
            "pred": pred_np,
            "hdim": int(hdim),
        })

        print({"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2),
               "RMSE_last": float(rmse_last), "MAE_last": float(mae_last), "R2_last": float(r2_last)})

    def find_index(self, array, flag):
        if flag == "min":
            retval = np.min(array)
            indx = np.argmin(array)
        if flag == "max":
            retval = np.max(array)
            indx = np.argmax(array)
        return retval, indx
