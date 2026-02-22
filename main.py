import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
#import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

#GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


DATA_PATHS = {
    "jphfmd": {"feat": "data/hfmd_num1.csv", "adj": "data/adj0.csv"},
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_model(args, dm):
    model = models.STGMixer(adj=dm.adj, hidden_dim=args.hidden_dim, pre_len=args.pre_len, seq_len=args.seq_len)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))  #  CUDA_VISIBLE_DEVICES  1， cuda:0
    return model

def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, feat_min_val=dm.feat_min_val, **vars(args)
    )
    return task

def get_callbacks(args):
    monitor = "val_r2"
    early_stop = EarlyStopping(
        monitor=monitor,
        min_delta=1e-3,  # 0.001
        patience=50,  # 50-70
        mode="max",
        verbose=True,
        check_finite=True
    )
    filename = ("HMF-{epoch:02d}-{train_loss:.2f}--" +
                ("{val_R2_last:.3f}" if monitor=="val_R2_last" else "{val_r2:.3f}"))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        dirpath=f"ckpt/pre_len{args.pre_len}",
        filename=filename,
        save_top_k=3,
        mode="max"
    )

    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    return [early_stop,checkpoint_callback, plot_validation_predictions_callback]

def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,deterministic=True, accelerator='gpu', devices=[0])
    trainer.fit(task, dm, ckpt_path=None)
    results = trainer.test(datamodule=dm, ckpt_path='best')

    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results

def load_model_from_ckpt(ckpt_path, model):
    checkpoint = torch.load(ckpt_path)

    model.load_state_dict(checkpoint['state_dict'])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data",
        type=str,
        help="The name of the dataset",
        choices=("jphfmd"),
        default="jphfmd"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("STGMixer"),
        default="STGMixer",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args()

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)

    args = parser.parse_args()
    pl.seed_everything(42, workers=True) #42
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)
    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()