import numpy as np
import mxnet as mx
from gluonts.mx.trainer.callback import *
import time
from utils.tools import copy_parameters
from evaluate.evaluate_forecast import get_agg_metrics
from pathlib import Path
from utils.tools import create_dir

class TrainingHistory(Callback):
    @validated()
    def __init__(self, path, dummy_predictor, validation_ds, monash, ctx):
        self.loss_history = []
        self.validation_loss_history = []
        self.epoch_no_history = []
        self.epoch_time_history = []
        self.lr_history = []
        self.gradient_stats={}
        self.model_parameters=None
        self.tolerance = 1e-4
        self.patience_limit = 10
        self.current_best = np.inf
        self.patience_counter = 0
        self.start_time = None
        self.print_str = ""
        self.save_path = path
        self.dummy_predictor = dummy_predictor
        self.validation_ds = validation_ds
        self.monash = monash
        self.val_metrics = {}
        self.ctx = mx.gpu() if ctx == "gpu" else mx.cpu()
    
    def count_model_params(self, net: nn.HybridBlock) -> int:
        params = net.collect_params()
        num_params = 0
        for p in params:
            v = params[p]
            num_params += np.prod(v.shape)
        return num_params

    def on_network_initializing_end(self, training_network: nn.HybridBlock) -> None:
        self.model_parameters = self.count_model_params(training_network)
        print("Number of parameters ", self.model_parameters)

    def on_train_start(self, max_epochs: int) -> None:
        self.start_time = time.time()
        print("time started")


    def on_train_epoch_end(self, epoch_no: int, epoch_loss: float, training_network: nn.HybridBlock,trainer: gluon.Trainer) -> bool:
        self.loss_history.append(epoch_loss)
        self.epoch_no_history.append(epoch_no)
        end = time.time()
        time_diff = end-self.start_time
        self.epoch_time_history.append(time_diff)
        lr = trainer.optimizer.learning_rate
        self.lr_history.append(lr)
        self.print_str = f"Epoch: {epoch_no} | time : {round(time_diff, 3)} | train_loss : {round(epoch_loss, 4)}"
        # if use gpu or not (by default the dummy predictor gets context as cpu as it is loaded from a file)

        
        copy_parameters(training_network, self.dummy_predictor.prediction_net, ctx=self.ctx)
        agg_metrics, final_forecast = get_agg_metrics(self.validation_ds, self.dummy_predictor, self.monash)
        for key, value in agg_metrics.items():
            if key not in self.val_metrics: self.val_metrics[key] = [] # intialize
            self.val_metrics[key].append(value)
        return True

    def on_validation_epoch_end(self,epoch_no: int,epoch_loss: float,training_network: nn.HybridBlock,trainer: gluon.Trainer) -> bool:
        
        grad_stat_keys = ["mean", "max", "median", "min", "std", "q10", "q20","q30","q40","q50","q60","q70","q80","q90","q99"]

        ## GET GRADIENT STATS
        for param in enumerate(training_network.collect_params()):
            # print(param, training_network.collect_params()[param[1]].data().asnumpy().shape) 
            if param[1] not in self.gradient_stats.keys(): 
                self.gradient_stats[param[1]]={key: [] for key in grad_stat_keys}

            # agg stats
            self.gradient_stats[param[1]]["mean"].append(np.nanmean(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1)))
            self.gradient_stats[param[1]]["max"].append(np.max(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1)))
            self.gradient_stats[param[1]]["median"].append(np.nanmedian(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1)))
            self.gradient_stats[param[1]]["min"].append(np.min(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1)))
            self.gradient_stats[param[1]]["std"].append(np.nanstd(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1)))
            

            self.gradient_stats[param[1]]["q10"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.1))
            self.gradient_stats[param[1]]["q20"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.2))
            self.gradient_stats[param[1]]["q30"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.3))
            self.gradient_stats[param[1]]["q40"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.4))
            self.gradient_stats[param[1]]["q50"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.5))
            self.gradient_stats[param[1]]["q60"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.6))
            self.gradient_stats[param[1]]["q70"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.7))
            self.gradient_stats[param[1]]["q80"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.8))
            self.gradient_stats[param[1]]["q90"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.9))
            self.gradient_stats[param[1]]["q99"].append(np.nanquantile(training_network.collect_params()[param[1]].data().asnumpy().reshape(-1), 0.99))
        
        ## GET 
        self.validation_loss_history.append(epoch_loss)
        self.print_str += f" | val_loss : {round(epoch_loss,4)}"
        print(self.print_str)

        if epoch_loss < self.current_best:
            self.current_best = epoch_loss
            self.patience_counter = 0
            create_dir(self.save_path+"~best")
            self.dummy_predictor.serialize(Path(self.save_path+"~best"))


        elif epoch_loss > self.current_best:
            self.patience_counter+=1
        

        # if len(self.validation_loss_history)>1 and np.abs(self.validation_loss_history[-1]-self.validation_loss_history[-2])<self.tolerance:
            # print("Early stopping based on validation loss")
        # if self.patience_counter>10:
        #     print("Early stopping based on validation loss; maximum patience reached")
        #     return False
        return True
        