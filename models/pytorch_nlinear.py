from re import X
import torch
import mxnet as mx
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gluonts
import torch
import torch.nn as nn
from typing import List,Callable
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.model.forecast_generator import DistributionForecastGenerator
import pytorch_lightning as pl
from gluonts.torch.modules.distribution_output import StudentTOutput
from gluonts.torch.modules.distribution_output import AffineTransform
from gluonts.torch.modules.distribution_output import NormalOutput
from gluonts.mx.trainer._base import Trainer
from evaluate.evaluate_forecast import get_agg_metrics
import time
import os
import numpy as np
from pathlib import Path
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)

class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, freq, prediction_length, 
                        context_length, 
                        nodes,
                        activation,
                        distribution_hidden_state,
                        scaling: Callable=mean_abs_scaling,
                        distr_output = StudentTOutput()):
        super(NLinear, self).__init__()

        self.seq_len = context_length
        self.context_length = context_length
        self.freq = freq
        self.pred_len = prediction_length
        self.prediction_length = prediction_length
        self.distribution_hidden_state = distribution_hidden_state
        self.distr_output = distr_output
        self.activation = activation
        self.probablistic = False
        self.nodes=nodes
        if 0 in self.nodes:
            self.nodes.remove(0)
        self.nodes[-1] = self.prediction_length * self.distribution_hidden_state
        print(self.nodes)
        self.n_layers = len(self.nodes)
        self.linear_dict = {}
        self.activation_dict = {}
        self.scaling = scaling
        self.nn_list = nn.ModuleList()

        for i in range(self.n_layers-1):
            self.linear_dict[i] = nn.Linear(self.nodes[i], self.nodes[i+1]) 
            self.activation_dict[i] = self.activation

            # add to nn list
            self.nn_list.append(self.linear_dict[i])
            self.nn_list.append(self.activation_dict[i])

        del self.nn_list[-1] # remove the last activation
        self.args_proj = self.distr_output.get_args_proj(self.distribution_hidden_state) # univariate
        self.nn = nn.Sequential(
            self.nn_list,
            self.args_proj)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        scale = self.scaling(x) # not used
        x=(x/scale)
        x = torch.unsqueeze(x, 2)
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        for i in range(self.n_layers-1):
            # pdb.set_trace()
            x = self.linear_dict[i](x.permute(0,2,1)).permute(0,2,1)
            if i <=self.n_layers-2:
                x = self.activation_dict[i](x)
            # x = x.permute(0,2,1)
        x = x + seq_last
        x = x.reshape(-1, self.pred_len, self.distribution_hidden_state)
        # x = torch.squeeze(x,2)
        distr_args = self.args_proj(x)
        # pdb.set_trace()
        return distr_args, torch.zeros_like(scale), scale


    def get_predictor(self, input_transform, batch_size=32):
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

class LightningNLinear(NLinear, pl.LightningModule):
    def __init__(self, validation_ds : ListDataset, 
                        monash, input_transform : gluonts.transform,
                        lr, patience, decay, retrain_run, rerun_epoch_max, integer_conversion, # loss,
                        model_save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_ds = validation_ds
        self.monash = monash
        self.input_transform = input_transform
        self.integer_conversion = integer_conversion
        # self.loss_criterion = nn.MSELoss() if loss == "mse" else nn.L1Loss()
        self.lr = lr
        self.patience = patience
        self.decay = decay
        self.training_step_loss = []
        self.loss_history= []
        self.val_step_loss = []
        self.validation_loss_history= []
        self.epoch_no = 0
        self.best_epoch_no = 0
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
        self.retrain_run = retrain_run
        self.rerun_epoch_max = rerun_epoch_max

    def on_train_start(self) -> None:
        self.model_parameters = sum(p.numel() for p in self.nn.parameters())
        self.start_time = time.time()
        print(self.nn)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        context = batch["past_target"].to(device)
        # pdb.set_trace()
        target = batch["future_target"].to(device)
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length
        distr_args, loc, scale = self(context)
        # pdb.set_trace()
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)
        self.training_step_loss.append(loss.mean())
        return loss.mean()

    def training_epoch_end(self, outputs) -> None:
        # train loss
        epoch_loss = sum(self.training_step_loss) / len(self.training_step_loss)
        self.loss_history.append(epoch_loss.item())
        
        # epoch_no
        self.epoch_no_history.append(self.epoch_no)
        self.epoch_no +=1

        # time
        end=time.time()
        time_diff = end-self.start_time
        self.epoch_time_history.append(time_diff)

        # reinitialize for next epoch
        self.training_step_loss = [] 
        
        self.print_str += f"\nEpoch: {self.epoch_no} | time : {round(time_diff, 3)} | train_loss : {np.round(epoch_loss.item(), 4)}"

        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        context = batch["past_target"].to(device)
        target = batch["future_target"].to(device)
        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length
        distr_args, loc, scale = self(context)
        # pdb.set_trace()
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)
        self.val_step_loss.append(loss.mean())
        self.log("val_loss", loss.mean())
        return loss.mean()


    def validation_epoch_end(self, outputs) -> None:
        # validation loss
        val_loss = sum(self.val_step_loss) / len(self.val_step_loss)
        self.validation_loss_history.append(val_loss.item())
        self.print_str += f" | val_loss : {np.round(val_loss.item(),4)}"
        print(self.print_str)
        self.print_str = ""

        # get lr
        self.lr_history.append(self.scheduler.get_last_lr()[0])

        # metrics per epoch
        predictor = self.get_predictor(self.input_transform)
        agg_metrics, final_forecast = get_agg_metrics(self.validation_ds, predictor, self.monash, self.integer_conversion, pytorch=True)
        for key, value in agg_metrics.items():
            if key not in self.val_metrics: self.val_metrics[key] = [] # intialize
            self.val_metrics[key].append(value)

        # model grad stats
        grad_stat_keys = ["mean", "max", "median", "min", "std", "q10", "q20","q30","q40","q50","q60","q70","q80","q90","q99"]
        for name, layer_parameter in self.nn.named_parameters():
            param = [layer_parameter, name]
            if param[1] not in self.gradient_stats.keys(): 
                self.gradient_stats[param[1]]={key: [] for key in grad_stat_keys}
            self.gradient_stats[param[1]]["mean"].append(torch.nanmean(layer_parameter.data.reshape(-1)).item())
            self.gradient_stats[param[1]]["max"].append(torch.max(layer_parameter.data.reshape(-1)).item())
            self.gradient_stats[param[1]]["median"].append(torch.nanmedian(layer_parameter.data.reshape(-1)).item())
            self.gradient_stats[param[1]]["min"].append(torch.min(layer_parameter.data.reshape(-1)).item())
            self.gradient_stats[param[1]]["std"].append(np.nanstd(layer_parameter.data.cpu().detach().numpy().reshape(-1)))
            
            self.gradient_stats[param[1]]["q10"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.1).item())
            self.gradient_stats[param[1]]["q20"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.2).item())
            self.gradient_stats[param[1]]["q30"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.3).item())
            self.gradient_stats[param[1]]["q40"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.4).item())
            self.gradient_stats[param[1]]["q50"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.5).item())
            self.gradient_stats[param[1]]["q60"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.6).item())
            self.gradient_stats[param[1]]["q70"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.7).item())
            self.gradient_stats[param[1]]["q80"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.8).item())
            self.gradient_stats[param[1]]["q90"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.9).item())
            self.gradient_stats[param[1]]["q99"].append(torch.nanquantile(layer_parameter.data.reshape(-1), 0.99).item())
        self.log("val_loss", val_loss)

        
        if val_loss < self.current_best:
            # save model state
            print(f"validation loss changed from {self.current_best} to {val_loss}")
            self.current_best = val_loss
            self.patience_counter = 0
            create_dir(self.model_save_path+"~best")
            if not self.retrain_run: # create a checkpoint only for the inital run
                torch.save(self.state_dict(), Path(self.model_save_path+"~best") / "best_checkpoint.pth")
                print("saving checkpoint ...")
                self.best_epoch_no = self.epoch_no
            else: # if re_run training ?
                if self.epoch_no == self.rerun_epoch_max: # if the epoch no is the last rerun epoch?
                    torch.save(self.state_dict(), Path(self.model_save_path+"~best") / "final_rerun_checkpoint.pth")
                    print("saving checkpoint ...")

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.patience, gamma = self.decay)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

def create_dir(directory):
    # create directory if does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)