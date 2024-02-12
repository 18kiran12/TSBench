import os
import gluonts
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.tft import TemporalFusionTransformerEstimator
from gluonts.mx.distribution.gaussian import GaussianOutput
from gluonts.mx.distribution.beta import BetaOutput 
from gluonts.mx.distribution.poisson import PoissonOutput 
from gluonts.mx.distribution.student_t import StudentTOutput

from gluonts.torch.modules.distribution_output import StudentTOutput
from gluonts.torch.modules.distribution_output import AffineTransform
from gluonts.torch.modules.distribution_output import NormalOutput

from models.simpleff import LightningNetwork
from models.pytorch_dlinear import LightningDLinear
from models.pytorch_nlinear import LightningNLinear

from torch import nn
from pathlib import Path
import tempfile
import mxnet as mx
import numpy as np
import pickle as pkl

from pkg_resources import get_distribution

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_model(args):
    model_map ={
        "ff": SimpleFeedForwardEstimator,
        "simpleff": LightningNetwork,
        "dlinear" : LightningDLinear,
        "nlinear" : LightningNLinear,
        "nbeats": NBEATSEstimator,
        "transformer": TransformerEstimator,
        "deepar": DeepAREstimator,
        "tft": TemporalFusionTransformerEstimator,
    }
    identifier = "model_"+args.model
    model_args = args[identifier]
    model_args = {k:v for k, v in model_args.items() if "empty" not in k}
    return model_map[args.model], model_args

def get_distribution(model_args : dict) -> dict:
    if "distr_output" in model_args: 
        dist_map = {
            "StudentTOutput" : StudentTOutput(),
            "GaussianOutput" : GaussianOutput(),
            "PoissonOutput" : PoissonOutput(),
            "BetaOutput" : BetaOutput()
        }
        model_args["distr_output"] = dist_map[model_args["distr_output"]]
    return model_args

def get_pytorch_distribution(model_args : dict) -> dict:
    if "distr_output" in model_args: 
        dist_map = {
            "StudentTOutput" : gluonts.torch.modules.distribution_output.StudentTOutput(),
            "GaussianOutput" : gluonts.torch.modules.distribution_output.NormalOutput(),
            "PoissonOutput" : gluonts.torch.modules.distribution_output.PoissonOutput(),
            "BetaOutput" : gluonts.torch.modules.distribution_output.BetaOutput(),
            "GammaOutput": gluonts.torch.modules.distribution_output.GammaOutput()
        }
        model_args["distr_output"] = dist_map[model_args["distr_output"]]
    return model_args

def copy_parameters(
    net_source: mx.gluon.Block,
    net_dest: mx.gluon.Block,
    ignore_extra: bool = False,
    allow_missing: bool = False,
    ctx = mx.current_context(),
) -> None:
    """
    Copies parameters from one network to another.

    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    ignore_extra
        Whether to ignore parameters from the source that are not
        present in the target.
    allow_missing
        Whether to allow additional parameters in the target not
        present in the source.
    """
    with tempfile.TemporaryDirectory(
        prefix="gluonts-estimator-temp-"
    ) as model_dir:
        model_dir_path = str(Path(model_dir) / "tmp_model")
        net_source.save_parameters(model_dir_path)
        net_dest.load_parameters(
            model_dir_path,
            ctx=ctx,
            allow_missing=allow_missing,
            ignore_extra=ignore_extra,
        )

def get_seasonality_frequency(frequency):
    """
    returns the seasonality according to gluon and the frequency
    """
    SEASONALITY_MAP = {
        "minutely": [1440, 10080, 525960],
        "10_minutes": [144, 1008, 52596],
        "half_hourly": [48, 336, 17532],
        "hourly": [24, 168, 8766],
        "daily": 7,
        "weekly": int(365.25/7),
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1
        }
    FREQUENCY_MAP = {
        "minutely": "1min",
        "10_minutes": "10min",
        "half_hourly": "30min",
        "hourly": "1H",
        "daily": "1D",
        "weekly": "1W",
        "monthly": "1M",
        "quarterly": "1Q",
        "yearly": "1Y"
        }
    return FREQUENCY_MAP[frequency], SEASONALITY_MAP[frequency]
    
def create_dir(directory):
    # create directory if does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

import pandas as pd


def return_new_time_by_addition(train_start_time, freq, offset):
    if freq=='1D':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(days=offset)
        print("in days: ", train_start_time, "->", offset,"->",new_time)     
        
    elif freq=='1H':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(hours=offset) 
        print("in hours: ", train_start_time, "->", offset,"->",new_time, "len: ")   
        
    elif freq=='1W':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(weeks=offset) 
        print("in weeks: ", train_start_time, "->", offset,"->",new_time, "len: ")   

    elif freq=='1M':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(months=offset)
        print("in months: ", train_start_time, "->", offset,"->",new_time, "len: ")   
        
    elif freq=='1Q':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(months=offset*3)
        print("in quarter: ", train_start_time, "->", offset,"->",new_time, "len: ")   
        
    elif freq=='1Y':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(years=offset)
        print("in years: ", train_start_time, "->", offset,"->",new_time, "len: ")   
        
    elif freq=='30min':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(minutes=offset*30)
        print("in 30min: ", train_start_time, "->", offset,"->",new_time, "len: ")   

    elif freq=='10min':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(minutes=offset*10)
        print("in 10min: ", train_start_time, "->", offset,"->",new_time, "len: ")   
        
    elif freq=='1min':
        new_time = pd.to_datetime(train_start_time, format="%Y-%m-%d %H-%M-%S") + pd.DateOffset(minutes=offset)
        print("in 1min: ", train_start_time, "->", offset,"->",new_time, "len: ")   
    
    return new_time

def save_data(data_args, args, predictor, final_forecasts, agg_metrics, history, tss, forecasts):

    print("Dataset parameters ", data_args)
    create_dir(args.run_folder+"/predictors/" + args.run_full_name)
    predictor.serialize(Path(args.run_folder+"/predictors/" + args.run_full_name))
    # tss_array, final_forecasts_array = np.asarray(tss), np.ravel(np.asarray(final_forecasts))
    # tss_forecast_array = np.ravel(tss_array[:, -data_args["prediction_length"]:, 0]) # just get the predictions
    # forecast_df = pd.DataFrame({'test_data' : tss_forecast_array, 'forecast': final_forecasts_array})
    # forecast_df.to_csv(args.run_folder+"/stats/" + args.run_full_name+"~forecast.csv", index=False)
    with open(args.run_folder+"/stats/"+args.run_full_name +"~agg_metrics.pickle", 'wb')as fp:
        pkl.dump(agg_metrics,fp)
    with open(args.run_folder+"/stats/"+args.run_full_name +"~history_val_loss.pickle", 'wb')as fp:
        pkl.dump(history.validation_loss_history,fp)
    with open(args.run_folder+"/stats/"+args.run_full_name +"~history_stats.pickle", 'wb')as fp:
        pkl.dump(history.gradient_stats,fp)
        # also save a pickle for easy retrieval
    with open(args.run_folder+"/stats/"+args.run_full_name +"~forecast.pickle", 'wb')as fp:
        pkl.dump({'test_data' : tss,
                'forecasts' : final_forecasts}, fp)


def update_dtype(config,  dtype, param_name):
    """
    dtypes can be int_dtypes, list_dtypes, date_dtypes, float_dtypes etc from the ini file
    param_name is the new parameter added to config file
    """
    curr = config.get("dtypes", dtype)
    curr += f"{param_name},"
    config.set("dtypes", dtype, curr)
    return config

# helper function
def set_param(config, section: str, params:dict, value) -> None:
    # check if value is empty
    name = params["name"]
    datatype = params["datatype"]
    config.set(section, name, value)
    config = update_dtype(config, datatype, name)
    return config