import numpy as np
import mxnet as mx
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import time
from utils.tools import copy_parameters
from evaluate.evaluate_forecast import get_agg_metrics
from pathlib import Path
from utils.tools import create_dir


class LR_logger(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, _):
        print("callback lr ", trainer.lr_scheduler_configs.get_lr)
        