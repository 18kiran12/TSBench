import gluonts
import torch
import torch.nn as nn
from typing import List,Callable
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast_generator import DistributionForecastGenerator
import pytorch_lightning as pl
from gluonts.torch.modules.distribution_output import StudentTOutput
from gluonts.mx.trainer._base import Trainer
from evaluate.evaluate_forecast import get_agg_metrics
import time
import os
import numpy as np
from pathlib import Path

def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        freq : str,
        prediction_length : int,
        context_length : int,
        hidden_dimensions=None,
        trainer = Trainer,
        distr_output = StudentTOutput(),
        batch_norm: bool=True,
        scaling: Callable=mean_abs_scaling,
    ) -> None:
        super().__init__()
        
        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0
        
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling
        
        
        dimensions = [context_length] + hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(self.__make_lin(dimensions[-1], prediction_length * hidden_dimensions[-1]))
        
        self.nn = nn.Sequential(*modules)
        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1])
        
    @staticmethod
    def __make_lin(dim_in, dim_out):
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin
    
    def forward(self, context):
        scale = self.scaling(context)
        scaled_context = (context / scale)
        nn_out = self.nn(scaled_context)
        nn_out_reshaped = nn_out.reshape(-1, self.prediction_length, self.hidden_dimensions[-1])
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros_like(scale), scale
    
    def get_predictor(self, input_transform, batch_size=32, device=None):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            freq=self.freq, 
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
            device=device,
        )

from gluonts.dataset.common import ListDataset
class LightningNetwork(FeedForwardNetwork, pl.LightningModule):
    def __init__(self, validation_ds : ListDataset, 
                        monash, 
                        input_transform : gluonts.transform,
                        model_save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_ds = validation_ds
        self.monash = monash
        self.input_transform = input_transform
        self.training_step_loss = []
        self.loss_history= []
        self.val_step_loss = []
        self.validation_loss_history= []
        self.epoch_no = 0
        self.start_time = None
        self.epoch_no_history = []
        self.epoch_time_history = []
        self.lr_history= []
        self.val_metrics = {}
        self.gradient_stats = {}
        self.current_best = np.inf
        self.print_str = ""
        self.model_parameters = None
        self.model_save_path = model_save_path

    def on_train_start(self) -> None:
        self.model_parameters = sum(p.numel() for p in self.nn.parameters())
        self.start_time = time.time()
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        context = batch["past_target"]
        target = batch["future_target"]
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length
        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)
        self.training_step_loss.append(loss.mean())
        return loss.mean()

    def training_epoch_end(self, outputs) -> None:
        # train loss
        epoch_loss = sum(self.training_step_loss) / len(self.training_step_loss)
        self.loss_history.append(epoch_loss)
        
        # epoch_no
        self.epoch_no_history.append(self.epoch_no)
        self.epoch_no +=1

        # time
        end=time.time()
        time_diff = end-self.start_time
        self.epoch_time_history.append(time_diff)

        # reinitialize for next epoch
        self.training_step_loss = [] 
        
        self.print_str += f"\nEpoch: {self.epoch_no} | time : {round(time_diff, 3)} | train_loss : {np.round(epoch_loss.detach().numpy(), 4)}"

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        context = batch["past_target"]
        target = batch["future_target"]
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length
        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)
        self.val_step_loss.append(loss.mean())
        return loss.mean()


    def validation_epoch_end(self, outputs) -> None:
        # validation loss
        val_loss = sum(self.val_step_loss) / len(self.val_step_loss)
        self.validation_loss_history.append(val_loss)
        self.print_str += f" | val_loss : {np.round(val_loss.detach().numpy(),4)}"
        print(self.print_str)
        self.print_str = ""

        # get lr
        self.lr_history.append(self.scheduler.get_last_lr())

        # metrics per epoch
        predictor = self.get_predictor(self.input_transform)
        agg_metrics, final_forecast = get_agg_metrics(self.validation_ds, predictor, self.monash, pytorch=True)
        for key, value in agg_metrics.items():
            if key not in self.val_metrics: self.val_metrics[key] = [] # intialize
            self.val_metrics[key].append(value)

        # model grad stats
        grad_stat_keys = ["mean", "max", "median", "min", "std", "q10", "q20","q30","q40","q50","q60","q70","q80","q90","q99"]
        for name, layer_parameter in self.nn.named_parameters():
            param = [layer_parameter, name]
            if param[1] not in self.gradient_stats.keys(): 
                self.gradient_stats[param[1]]={key: [] for key in grad_stat_keys}
            self.gradient_stats[param[1]]["mean"].append(torch.nanmean(layer_parameter.data.reshape(-1)))
            self.gradient_stats[param[1]]["max"].append(torch.max(layer_parameter.data.reshape(-1)))
            self.gradient_stats[param[1]]["median"].append(torch.nanmedian(layer_parameter.data.reshape(-1)))
            self.gradient_stats[param[1]]["min"].append(torch.min(layer_parameter.data.reshape(-1)))
            self.gradient_stats[param[1]]["std"].append(np.nanstd(layer_parameter.data.cpu().detach().numpy().reshape(-1)))
            
            self.gradient_stats[param[1]]["q10"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.1))
            self.gradient_stats[param[1]]["q20"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.2))
            self.gradient_stats[param[1]]["q30"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.3))
            self.gradient_stats[param[1]]["q40"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.4))
            self.gradient_stats[param[1]]["q50"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.5))
            self.gradient_stats[param[1]]["q60"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.6))
            self.gradient_stats[param[1]]["q70"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.7))
            self.gradient_stats[param[1]]["q80"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.8))
            self.gradient_stats[param[1]]["q90"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.9))
            self.gradient_stats[param[1]]["q99"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.99))
        self.log("val_loss", val_loss)

        # save model state
        print("val_loss, self.current_best", val_loss, self.current_best)
        if val_loss < self.current_best:
            self.current_best = val_loss
            self.patience_counter = 0
            # create_dir(self.model_save_path+"~best")
            torch.save(self.state_dict(), Path(self.model_save_path+"~best") / "best_checkpoint.pth")

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 10, gamma =0.5)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

def create_dir(directory):
    # create directory if does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)