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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    """
    DLinear
    """
    def __init__(self, freq, prediction_length, 
                        context_length, 
                        kernel_size,
                        blocks,
                        scaling: Callable=mean_abs_scaling,
                        distr_output = StudentTOutput(),
                        ):
        super(DLinearModel, self).__init__()
        self.seq_len = context_length
        self.context_length = context_length
        self.freq = freq
        self.pred_len = prediction_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.probablistic = False
        self.blocks=blocks
        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        self.individual = False
        self.channels = 1
        self.scaling=scaling

        self.pre_nodes = {}


        
        self.Linear_Seasonal={}
        self.Linear_Trend={}
        self.Linear_Decoder={}
        self.nn_list = nn.ModuleList()

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            self.Linear_Decoder = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                self.Linear_Decoder.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            for i in range(self.blocks, self.nodes):
                if i==0:
                    self.Linear_Seasonal[i] = nn.Linear(self.seq_len,self.pred_len)
                    self.Linear_Trend[i] = nn.Linear(self.seq_len,self.pred_len)
                    self.Linear_Decoder[i] = nn.Linear(self.seq_len,self.pred_len)
                    self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                    self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                    self.nn_list.append(self.Linear_Seasonal[i])
                    self.nn_list.append(self.Linear_Trend[i])
                    # self.nn_list.append(self.Linear_Decoder[i])
                else:
                    self.Linear_Seasonal[i] = nn.Linear(self.pred_len,self.pred_len)
                    self.Linear_Trend[i] = nn.Linear(self.pred_len,self.pred_len)
                    self.Linear_Decoder[i] = nn.Linear(self.pred_len,self.pred_len)
                    self.Linear_Seasonal[i].weight = nn.Parameter((1/self.pred_len)*torch.ones([self.pred_len,self.pred_len]))
                    self.Linear_Trend[i].weight = nn.Parameter((1/self.pred_len)*torch.ones([self.pred_len,self.pred_len]))
                    self.nn_list.append(self.Linear_Seasonal[i])
                    self.nn_list.append(self.Linear_Trend[i])
                    # self.nn_list.append(self.Linear_Decoder[i])


        self.args_proj = self.distr_output.get_args_proj(1)

        self.nn = nn.Sequential(
            *self.nn_list,
            self.args_proj)

        # self.pre_linear_layers = {}
        # self.pre_linear_layers[0] = nn.Linear(self.seq_len, self.nodes[0])
        
        # for i, nodes in enumerate(self.nodes):
        #     self.pre_linear_layers[i] = nn.Linear(self.nodes[i], self.nodes[i+1])




    def forward(self, x):
        # x: [Batch, Input length, Channel]
        scale = self.scaling(x)
        x = (x / scale)
        x = torch.unsqueeze(x, 2)
        xs = []


        for i in range(self.blocks):
            # first iteration seq len
            seasonal_init, trend_init = self.decompsition(x) 
            seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            if self.individual:
                seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
                trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
                for i in range(self.channels):
                    seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                    trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
            else:
                seasonal_output = self.Linear_Seasonal[i](seasonal_init)
                trend_output = self.Linear_Trend[i](trend_init)

            x = seasonal_output + trend_output
            x = x.permute(0,2,1) # to [Batch, Output length, Channel]
            xs.append(x)
        out_x = torch.zeros_like(xs[0]) # create an empty torch tensor of the same shape as output
        # print(out_x[0,0,0])
        for xi in xs:
            out_x += xi
        # print(out_x[0,0,0], out_x.shape)
        distr_args = self.args_proj(out_x)
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
class LightningDLinear(DLinearModel, pl.LightningModule):
    def __init__(self, validation_ds : ListDataset, 
                        monash, input_transform : gluonts.transform,
                        lr, patience, decay,
                        model_save_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_ds = validation_ds
        self.monash = monash
        self.input_transform = input_transform
        self.loss_criterion = nn.MSELoss()
        self.lr = lr
        self.patience = patience
        self.decay = decay
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
        print(self.nn)
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        context = batch["past_target"].to(device)
        target = batch["future_target"].to(device)
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
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)
        self.val_step_loss.append(loss.mean())
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
            torch.save(self.state_dict(), Path(self.model_save_path+"~best") / "best_checkpoint.pth")

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.patience, gamma = self.decay)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

def create_dir(directory):
    # create directory if does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)