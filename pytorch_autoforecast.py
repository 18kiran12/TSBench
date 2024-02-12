from logging import raiseExceptions
import torch
import pytorch_lightning as pl
from matplotlib.pyplot import hist
import mxnet as mx
import numpy as np
import pandas as pd
import json
import argparse
from configparser import ConfigParser
from datetime import datetime,timedelta
from mxnet.callback import do_checkpoint
import pdb
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.loader import ValidationDataLoader
from gluonts.dataset.util import to_pandas
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
import inspect
import os
from torch.nn import ELU, ReLU, LeakyReLU

from pathlib import Path
import pickle
from visualize.plot import plot_train_test_split, plot_prob_forecasts
from dataset.custom_dataset import artificial, Monash_Dataset
from utils.parse_config import ParseConfig
from utils.tools import get_model, create_dir, save_data, get_pytorch_distribution
from evaluate.evaluate_forecast import get_agg_metrics
from dataset.pytorch_dataset import get_context_data_loader
from models.simpleff import LightningNetwork
from models.pytorch_callbacks import LR_logger
import warnings

warnings.filterwarnings("ignore")
plot = False
import sys

import random


parser = argparse.ArgumentParser(description='Autoforecasting for time series')
parser.add_argument('--config_file', type=str, default='nlinear~australian_electricity_demand_dataset~0a6d26a4-e0e0-43e9-8c01-313777d56f0c~2.ini')
parser.add_argument('--itr', type=int, default=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"usng device {device}")
##################### get config ###############################
args_config_file = parser.parse_args()

config = ConfigParser()
# check if config file exist
my_file = Path(args_config_file.config_file)
if my_file.is_file():
    print("Config file found")
else:
    raise FileNotFoundError
config.read(args_config_file.config_file)
pc = ParseConfig(config)
args = pc.as_dict()

print(args)
args.run_full_name = args.model+"~"+args.dataset.split('.')[0] + "~" +args.run_name+ "~" +str(args_config_file.itr)
# model_save_path = args.run_folder+"/predictors/"+args.run_full_name
model_save_path = args.run_folder+"/predictors/"+args.run_full_name

## set seeds
random.seed(args.seed)
mx.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

##################### get dataset ##############################


# custom dataset - MONASH
monash = Monash_Dataset(args)
train_ds, test_ds, val_ds, train_val_ds, data_args = monash.get_monash_dataset()
# if plot: plot_train_test_split(train_ds, val_ds, test_ds, args)
# entry = next(iter(train_val_ds))
# print("train_val_ds", entry, "\n len:", train_val_ds.__len__())

# entry = next(iter(train_ds))
# print("train_ds",entry, "\n len:", train_ds.__len__())

# entry = next(iter(test_ds))
# print("test_ds", entry, "\n len:", test_ds.__len__())

##################### train ####################################
start_exec_time = datetime.now()

model, model_args = get_model(args)
model_args = get_pytorch_distribution(model_args) # update model args with distribution function

ctx=-1 if args.ctx == "gpu" else 0

data_dict = {
    "train_ts" : train_ds,
    "val_ts": val_ds,
    "test_ts" : test_ds,
    "train_val_ts" : train_val_ds,
    "freq" : data_args["freq"],
    "context_length" : args["context_length"], # context length passed as argument
    "prediction_length" : data_args["prediction_length"],
    "batch_size" : args.model_trainer["batch_size"],
    "num_batches_per_epoch" : args.model_trainer["num_batches_per_epoch"]
}

train_dl, val_dl, test_dl, train_val_dl, transform = get_context_data_loader(**data_dict)

# train_val_dl = [d for dl in [train_dl, val_dl] for d in dl] # did not work max recursion error

# entry = next(iter(train_val_dl))
# print("trian_val_dl", entry["future_target"])

# entry = next(iter(train_dl))
# print("trian_dl",entry["future_target"])

model, model_args = get_model(args) # reinitialize
model_args = get_pytorch_distribution(model_args)

if "nodes" in model_args.keys(): # work around to fix checkpoint issue
    model_args["nodes"].insert(0, args["context_length"])
    model_args["nodes"].append(data_args["prediction_length"])

if "activation" in model_args.keys():
    if model_args["activation"] == "elu":
        model_args["activation"] = ELU()
    else:
        model_args["activation"] = LeakyReLU()

print(model_args)

estimator = model(
    model_save_path=model_save_path,
    validation_ds = val_ds, 
    monash = monash,
    input_transform = transform,
    lr = args.model_trainer["learning_rate"],
    patience = args.model_trainer["patience"],
    decay = args.model_trainer["decay"],
    # loss = args.model_trainer["loss"],
    retrain_run = False,
    rerun_epoch_max = 0,
    integer_conversion = args.integer_conversion,
    # general dataset parameters
    freq=data_args["freq"],
    prediction_length=data_args["prediction_length"],
    context_length = args["context_length"],
    # estimator parameters
    **model_args
)

trainer = pl.Trainer(max_epochs=args.model_trainer["epochs"], gpus=-1 if torch.cuda.is_available() else None )
trainer.fit(estimator,  train_dataloaders=train_dl, val_dataloaders=val_dl)

# Execution time
finish_exec_time = datetime.now()
exec_time = finish_exec_time - start_exec_time
print("Execution time : ", exec_time)


##################### forecast ####################################
best_estimator = model(
    model_save_path=model_save_path,
    validation_ds = val_ds, 
    monash = monash,
    input_transform = transform,
    lr = args.model_trainer["learning_rate"],
    patience = args.model_trainer["patience"],
    decay = args.model_trainer["decay"],
    # loss = args.model_trainer["loss"],
    retrain_run = True,
    rerun_epoch_max = estimator.best_epoch_no,
    integer_conversion = args.integer_conversion,
    # general dataset parameters
    freq=data_args["freq"],
    prediction_length=data_args["prediction_length"],
    context_length = args["context_length"],
    # estimator parameters
    **model_args
)

best_estimator.load_state_dict(torch.load(Path(model_save_path+"~best") / "best_checkpoint.pth"))

predictor_pytorch = best_estimator.get_predictor(transform)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor_pytorch,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

##################### Evaluate ####################################

forecasts = list(f.to_sample_forecast() for f in forecast_it)
tss = list(ts_it)

# Get median (0.5 quantile) of the 100 sample forecasts as final point forecasts
if plot: plot_prob_forecasts(tss, forecasts, args)

print("Evaluating")

agg_metrics, final_forecasts = get_agg_metrics(test_ds, predictor_pytorch, monash, args.integer_conversion, pytorch=True)
##################### Save useful data #################################

forecast_dict = {}
for i, pred in enumerate(final_forecasts):
    forecast_dict[i] = pred.tolist()

results_dict = {"time" :  estimator.epoch_time_history,
                "epoch" : estimator.epoch_no_history,
                "train_loss": estimator.loss_history,
                "validation_loss" : estimator.validation_loss_history,
                "val_metrics" : estimator.val_metrics,
                "grad_stats": estimator.gradient_stats,
                "lr":estimator.lr_history,
                "model_parameters": estimator.model_parameters,
}
args["execution_time"] = str(exec_time)

json_data = {**results_dict, **agg_metrics, **data_args, **args}

with open(args.run_folder+"/json_metadata/"+args.run_full_name+".json", 'w') as f:
    json.dump(str(json_data), f)
with open(args.run_folder+"/json_metadata/"+args.run_full_name+"~forecast.json", 'w') as f:
    json.dump(str(forecast_dict), f)

with open(args.run_folder+"/json_metadata/"+args.run_full_name+".pkl", 'wb') as f:
    pickle.dump(json_data, f)

if args["validation_strategy"] == "re_oos":
    # retrain the model to predifined number of steps
    retrain_start_exec_time = datetime.now()
    epoch_no = estimator.best_epoch_no
    print("retrain until epoch ", epoch_no)


    best_estimator = model(
        model_save_path=model_save_path,
        validation_ds = val_ds, 
        monash = monash,
        input_transform = transform,
        lr = args.model_trainer["learning_rate"],
        patience = args.model_trainer["patience"],
        decay = args.model_trainer["decay"],
        # loss = args.model_trainer["loss"],
        retrain_run = True,
        rerun_epoch_max = estimator.best_epoch_no,
        integer_conversion = args.integer_conversion,
        # general dataset parameters
        freq=data_args["freq"],
        prediction_length=data_args["prediction_length"],
        context_length = args["context_length"],
        # estimator parameters
        **model_args
    )

    trainer = pl.Trainer(max_epochs=epoch_no, gpus=-1 if torch.cuda.is_available() else None )
    trainer.fit(best_estimator, train_dataloaders=train_val_dl, val_dataloaders=val_dl) # train the best estimator  
    predictor_pytorch = best_estimator.get_predictor(transform)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_ds,  # test dataset
        predictor=predictor_pytorch,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    ##################### Evaluate ####################################

    forecasts = list(f.to_sample_forecast() for f in forecast_it)
    tss = list(ts_it)

    # Get median (0.5 quantile) of the 100 sample forecasts as final point forecasts
    if plot: plot_prob_forecasts(tss, forecasts, args)

    print("Evaluating")

    agg_metrics, final_forecasts = get_agg_metrics(test_ds, predictor_pytorch, monash, args.integer_conversion, pytorch=True)
    ##################### Save useful data #################################

    forecast_dict = {}
    for i, pred in enumerate(final_forecasts):
        forecast_dict[i] = pred.tolist()



    retrain_finish_exec_time = datetime.now()
    retrain_exec_time = retrain_finish_exec_time - retrain_start_exec_time
    print("Execution time : ", retrain_exec_time)

    results_dict = {"time" :  best_estimator.epoch_time_history,
                    "epoch" : best_estimator.epoch_no_history,
                    "train_loss": best_estimator.loss_history,
                    "validation_loss" : best_estimator.validation_loss_history,
                    "val_metrics" : best_estimator.val_metrics,
                    "grad_stats": best_estimator.gradient_stats,
                    "lr":best_estimator.lr_history,
                    "model_parameters": best_estimator.model_parameters,
    }
    args["execution_time"] = str(retrain_exec_time)

    json_data = {**results_dict, **agg_metrics, **data_args, **args}

    with open(args.run_folder+"/json_metadata/"+args.run_full_name+"~retrain.json", 'w') as f:
        json.dump(str(json_data), f)
    with open(args.run_folder+"/json_metadata/"+args.run_full_name+"~retrain~forecast.json", 'w') as f:
        json.dump(str(forecast_dict), f)

    with open(args.run_folder+"/json_metadata/"+args.run_full_name+"~retrain.pkl", 'wb') as f:
        pickle.dump(json_data, f)   