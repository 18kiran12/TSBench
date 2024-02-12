# TSBench
Hyperparameter Tuning MLP's for Probabilistic Time Series Forecasting
 
TSBench focuses on examining the impact of specific hyperparameters related to time series, such as context length and validation strategy, on the performance of the state-of-the-art MLP model in time series forecasting. 

## Dataset

We use the dataset from [Monash Forecasting Repository](https://forecastingdata.org/). Once the datasets are downloaded, the used can define network parameters and dataset parameters using `configs/datasets.json` and `model_parameters.json` files, respectively.

## Run Configuration

The run configuration can be defined by using an `ini` file as shown in `nlinear~australian_electricity_demand_dataset~0a6d26a4-e0e0-43e9-8c01-313777d56f0c~2.ini`. The ini file has various hyperparameter configurations including the validation strategy, model, epochs, learning rate, etc. Each `ini` file corresponds to a single hyperparameter run. For our experiments, we create, for each hyperparameter run a unique `ini` file identified by the run name `0a6d26a4-e0e0-43e9-8c01-313777d56f0c`. Inorder to run a specific configuration.
```python
python pytorch_autoforecast.py --config_file nlinear~australian_electricity_demand_dataset~0a6d26a4-e0e0-43e9-8c01-313777d56f0c~2.ini --itr 0
```
The `itr` argument is for running the same configuration multiple times. 



